"""
LIVE BATTLEFIELD SIMULATION
===========================
$100 capital. Real market data. Real timestamps. Real prices.
Every trade verifiable against actual exchange/market records.

NOT real execution — but indistinguishable from it in reporting accuracy.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config.loader import load_settings
from data.ingestion.crypto import CryptoProvider
from data.ingestion.indian_equity import IndianEquityProvider
from data.ingestion.us_equity import USEquityProvider
from indicators.technical import compute_all
from risk.engine import RiskEngine, PortfolioState
from data.models import Signal, TradeSignal, BacktestResult
from strategy.modules.trend_follower import TrendFollower
from strategy.modules.mean_reverter import MeanReverter
from strategy.modules.breakout import BreakoutDetector


INITIAL_CAPITAL = 100.0
TARGET = 120.0


class BattlefieldSimulation:
    """Real-data paper trading simulation with audit-grade logging."""

    def __init__(self):
        self.settings = load_settings()
        self.risk = RiskEngine(self.settings.risk)
        self.risk.state.equity = INITIAL_CAPITAL
        self.risk.state.peak_equity = INITIAL_CAPITAL

        # Providers
        self.crypto = CryptoProvider("binance", self.settings.engine.data_dir)
        self.us_equity = USEquityProvider(self.settings.engine.data_dir)
        self.indian_eq = IndianEquityProvider(self.settings.engine.data_dir)

        # Strategy modules
        self.modules = [TrendFollower(), MeanReverter(), BreakoutDetector()]

        # Trade journal — every action recorded
        self.journal: list[dict] = []
        self.equity_curve: list[dict] = []
        self.signals_generated = 0
        self.signals_rejected = 0
        self.trades_executed = 0

    async def fetch_market_data(self) -> dict[str, pd.DataFrame]:
        """Fetch real market data for simulation universe."""
        datasets = {}
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=365)

        # Crypto — BTC and ETH via yfinance fallback
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            try:
                df = await self.crypto.get_ohlcv(symbol, "1d", start, now)
                if len(df) > 50:
                    datasets[symbol] = df
                    self._log_event("DATA_LOADED", {
                        "symbol": symbol,
                        "bars": len(df),
                        "from": str(df.index[0]),
                        "to": str(df.index[-1]),
                        "last_close": round(float(df["close"].iloc[-1]), 2),
                        "source": "yfinance_fallback" if "binance" not in str(type(df)) else "ccxt_binance",
                    })
            except Exception as e:
                self._log_event("DATA_ERROR", {"symbol": symbol, "error": str(e)})

        # US equities
        for symbol in ["AAPL", "NVDA", "SPY"]:
            try:
                df = await self.us_equity.get_ohlcv(symbol, "1d", start, now)
                if len(df) > 50:
                    datasets[symbol] = df
                    self._log_event("DATA_LOADED", {
                        "symbol": symbol,
                        "bars": len(df),
                        "from": str(df.index[0]),
                        "to": str(df.index[-1]),
                        "last_close": round(float(df["close"].iloc[-1]), 2),
                        "source": "yfinance",
                    })
            except Exception as e:
                self._log_event("DATA_ERROR", {"symbol": symbol, "error": str(e)})

        # Indian equities
        for symbol in ["RELIANCE", "TCS", "INFY"]:
            try:
                df = await self.indian_eq.get_ohlcv(symbol, "1d", start, now)
                if len(df) > 50:
                    datasets[symbol] = df
                    self._log_event("DATA_LOADED", {
                        "symbol": symbol,
                        "bars": len(df),
                        "from": str(df.index[0]),
                        "to": str(df.index[-1]),
                        "last_close": round(float(df["close"].iloc[-1]), 2),
                        "source": "jugaad_data_nse",
                    })
            except Exception as e:
                self._log_event("DATA_ERROR", {"symbol": symbol, "error": str(e)})

        return datasets

    def run_simulation(self, datasets: dict[str, pd.DataFrame]):
        """Walk through last 90 days bar-by-bar, simulating live trading."""

        self._log_event("SIMULATION_START", {
            "capital": INITIAL_CAPITAL,
            "target": TARGET,
            "target_return_pct": f"{((TARGET - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.1f}%",
            "assets": list(datasets.keys()),
            "strategy_modules": [m.name for m in self.modules],
            "risk_max_drawdown": f"{self.settings.risk.max_drawdown_pct}%",
            "risk_max_position": f"{self.settings.risk.max_position_pct}%",
            "commission_pct": 0.1,
            "slippage_pct": 0.05,
        })

        # Enrich all datasets with indicators
        enriched = {}
        for symbol, df in datasets.items():
            edf = compute_all(df)
            edf.attrs["symbol"] = symbol
            enriched[symbol] = edf

        # Find common date range (last 90 trading days)
        all_dates = set()
        for df in enriched.values():
            all_dates.update(df.index)
        sorted_dates = sorted(all_dates)
        sim_dates = sorted_dates[-90:]  # Last 90 trading days

        self._log_event("SIMULATION_WINDOW", {
            "start": str(sim_dates[0]),
            "end": str(sim_dates[-1]),
            "trading_days": len(sim_dates),
        })

        # Record initial equity
        self.equity_curve.append({
            "date": str(sim_dates[0]),
            "equity": INITIAL_CAPITAL,
            "drawdown_pct": 0.0,
            "open_positions": 0,
        })

        # Day-by-day simulation
        for day_idx, current_date in enumerate(sim_dates):
            daily_signals = []

            # Check stops on existing positions
            for symbol in list(self.risk.state.open_positions.keys()):
                if symbol not in enriched:
                    continue
                df = enriched[symbol]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]

                stop_hit = self.risk.check_stops(symbol, float(row["high"]), float(row["low"]))
                if stop_hit:
                    pos = self.risk.state.open_positions[symbol]
                    exit_price = pos.stop_loss if stop_hit == "sl" else pos.take_profit

                    # Apply slippage
                    slip = exit_price * 0.0005
                    exit_price = exit_price + slip if pos.side == Signal.BUY else exit_price - slip

                    pnl = self.risk.close_position(symbol, exit_price)
                    # Deduct commission
                    commission = abs(pos.notional) * 0.001
                    self.risk.state.equity -= commission
                    pnl -= commission

                    self.trades_executed += 1
                    self._log_event("TRADE_CLOSE", {
                        "trade_id": f"T{self.trades_executed:04d}",
                        "date": str(current_date),
                        "symbol": symbol,
                        "side": pos.side.value,
                        "entry_price": round(pos.entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "size": round(abs(pos.size), 6),
                        "pnl_usd": round(pnl, 4),
                        "pnl_pct": round(pnl / INITIAL_CAPITAL * 100, 4),
                        "exit_reason": "STOP_LOSS" if stop_hit == "sl" else "TAKE_PROFIT",
                        "commission_usd": round(commission, 4),
                        "equity_after": round(self.risk.state.equity, 4),
                        "module": pos.module_name,
                    })

            # Generate signals from all modules on all assets
            for symbol, df in enriched.items():
                if current_date not in df.index:
                    continue

                # Get data up to current date (no lookahead)
                historical = df.loc[:current_date]
                if len(historical) < 50:
                    continue

                for module in self.modules:
                    signals = module.generate_signals(historical)
                    if not signals:
                        continue

                    # Only take latest signal (today)
                    latest = signals[-1]
                    if pd.Timestamp(latest.timestamp) != current_date:
                        continue

                    # Fix symbol reference
                    latest = TradeSignal(
                        timestamp=latest.timestamp,
                        symbol=symbol,
                        signal=latest.signal,
                        confidence=latest.confidence,
                        entry_price=latest.entry_price,
                        stop_loss=latest.stop_loss,
                        take_profit=latest.take_profit,
                        module_name=latest.module_name,
                        reasoning=latest.reasoning,
                    )
                    daily_signals.append(latest)
                    self.signals_generated += 1

            # Process signals through risk engine
            for sig in daily_signals:
                if sig.signal == Signal.HOLD:
                    continue

                approved, reason, size = self.risk.approve_signal(sig)

                if not approved:
                    self.signals_rejected += 1
                    self._log_event("SIGNAL_REJECTED", {
                        "date": str(current_date),
                        "symbol": sig.symbol,
                        "signal": sig.signal.value,
                        "module": sig.module_name,
                        "confidence": round(sig.confidence, 3),
                        "reason": reason,
                    })
                    continue

                # Apply slippage to entry
                slip = sig.entry_price * 0.0005
                entry_price = sig.entry_price + slip if sig.signal == Signal.BUY else sig.entry_price - slip

                # Deduct commission
                commission = abs(size * entry_price) * 0.001
                self.risk.state.equity -= commission

                # Open position
                self.risk.open_position(sig, size)
                self.trades_executed += 1

                self._log_event("TRADE_OPEN", {
                    "trade_id": f"T{self.trades_executed:04d}",
                    "date": str(current_date),
                    "symbol": sig.symbol,
                    "side": sig.signal.value,
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(sig.stop_loss, 2),
                    "take_profit": round(sig.take_profit, 2),
                    "size": round(size, 6),
                    "notional_usd": round(abs(size * entry_price), 4),
                    "confidence": round(sig.confidence, 3),
                    "commission_usd": round(commission, 4),
                    "risk_per_trade_usd": round(abs(entry_price - sig.stop_loss) * size, 4),
                    "equity_after": round(self.risk.state.equity, 4),
                    "module": sig.module_name,
                    "reasoning": sig.reasoning,
                })

            # Record equity at end of day
            self.equity_curve.append({
                "date": str(current_date),
                "equity": round(self.risk.state.equity, 4),
                "drawdown_pct": round(self.risk.state.current_drawdown_pct, 4),
                "open_positions": len(self.risk.state.open_positions),
            })

        # Close remaining positions at last available price
        for symbol in list(self.risk.state.open_positions.keys()):
            if symbol in enriched:
                df = enriched[symbol]
                last_close = float(df["close"].iloc[-1])
                pos = self.risk.state.open_positions[symbol]
                pnl = self.risk.close_position(symbol, last_close)
                commission = abs(pos.notional) * 0.001
                self.risk.state.equity -= commission
                pnl -= commission
                self.trades_executed += 1
                self._log_event("TRADE_CLOSE", {
                    "trade_id": f"T{self.trades_executed:04d}",
                    "date": str(sim_dates[-1]),
                    "symbol": symbol,
                    "side": pos.side.value,
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(last_close, 2),
                    "size": round(abs(pos.size), 6),
                    "pnl_usd": round(pnl, 4),
                    "exit_reason": "SIMULATION_END",
                    "equity_after": round(self.risk.state.equity, 4),
                    "module": pos.module_name,
                })

    def generate_report(self) -> str:
        """Generate audit-grade simulation report."""
        now = datetime.now(timezone.utc)

        # Extract trade entries
        opens = [j for j in self.journal if j["event"] == "TRADE_OPEN"]
        closes = [j for j in self.journal if j["event"] == "TRADE_CLOSE"]
        rejected = [j for j in self.journal if j["event"] == "SIGNAL_REJECTED"]

        # P&L calculations
        trade_pnls = [c["data"]["pnl_usd"] for c in closes if "pnl_usd" in c["data"]]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        total_pnl = sum(trade_pnls)
        final_equity = self.risk.state.equity
        total_return_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        # Equity curve stats
        equities = [e["equity"] for e in self.equity_curve]
        peak = max(equities) if equities else INITIAL_CAPITAL
        trough = min(equities) if equities else INITIAL_CAPITAL
        max_dd = (peak - trough) / peak * 100 if peak > 0 else 0

        # Win/loss by module
        module_stats = {}
        for c in closes:
            mod = c["data"].get("module", "unknown")
            if mod not in module_stats:
                module_stats[mod] = {"wins": 0, "losses": 0, "pnl": 0.0}
            pnl = c["data"].get("pnl_usd", 0)
            module_stats[mod]["pnl"] += pnl
            if pnl > 0:
                module_stats[mod]["wins"] += 1
            else:
                module_stats[mod]["losses"] += 1

        lines = []
        lines.append("=" * 70)
        lines.append("  BATTLEFIELD SIMULATION REPORT")
        lines.append(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("  PARAMETERS")
        lines.append(f"  Initial Capital:     ${INITIAL_CAPITAL:.2f}")
        lines.append(f"  Target:              ${TARGET:.2f} ({((TARGET-INITIAL_CAPITAL)/INITIAL_CAPITAL)*100:.0f}% return)")
        lines.append(f"  Commission:          0.10% per side")
        lines.append(f"  Slippage:            0.05% per fill")
        lines.append(f"  Max Drawdown Limit:  {self.settings.risk.max_drawdown_pct}%")
        lines.append(f"  Max Position Size:   {self.settings.risk.max_position_pct}% of equity")
        lines.append("")

        # Simulation window
        if self.equity_curve:
            lines.append("  SIMULATION WINDOW")
            lines.append(f"  Start:  {self.equity_curve[0]['date']}")
            lines.append(f"  End:    {self.equity_curve[-1]['date']}")
            lines.append(f"  Days:   {len(self.equity_curve)}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("  RESULTS")
        lines.append("-" * 70)
        lines.append(f"  Final Equity:        ${final_equity:.2f}")
        lines.append(f"  Total P&L:           ${total_pnl:+.2f}")
        lines.append(f"  Total Return:        {total_return_pct:+.2f}%")
        lines.append(f"  Target Achieved:     {'YES' if final_equity >= TARGET else 'NO'}")
        lines.append("")
        lines.append(f"  Signals Generated:   {self.signals_generated}")
        lines.append(f"  Signals Rejected:    {self.signals_rejected}")
        lines.append(f"  Trades Executed:     {len(closes)}")
        lines.append(f"  Wins:                {len(wins)}")
        lines.append(f"  Losses:              {len(losses)}")
        lines.append(f"  Win Rate:            {len(wins)/len(trade_pnls)*100:.1f}%" if trade_pnls else "  Win Rate:            N/A")
        lines.append(f"  Avg Win:             ${sum(wins)/len(wins):+.4f}" if wins else "  Avg Win:             N/A")
        lines.append(f"  Avg Loss:            ${sum(losses)/len(losses):+.4f}" if losses else "  Avg Loss:            N/A")
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        lines.append(f"  Profit Factor:       {pf:.2f}")
        lines.append(f"  Max Drawdown:        {max_dd:.2f}%")
        lines.append(f"  Peak Equity:         ${peak:.2f}")
        lines.append("")

        # Module breakdown
        lines.append("-" * 70)
        lines.append("  MODULE PERFORMANCE")
        lines.append("-" * 70)
        for mod, stats in module_stats.items():
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total * 100 if total > 0 else 0
            lines.append(f"  {mod:20s}  trades={total:3d}  W/L={stats['wins']}/{stats['losses']}  "
                         f"WR={wr:.0f}%  PnL=${stats['pnl']:+.4f}")
        lines.append("")

        # Trade journal
        lines.append("-" * 70)
        lines.append("  TRADE JOURNAL (Every entry verifiable against market data)")
        lines.append("-" * 70)

        for entry in self.journal:
            if entry["event"] in ("TRADE_OPEN", "TRADE_CLOSE"):
                d = entry["data"]
                if entry["event"] == "TRADE_OPEN":
                    lines.append(
                        f"  {d.get('trade_id','')} | {d['date']} | OPEN  {d['side']:4s} "
                        f"{d['symbol']:12s} @ ${d['entry_price']:<10.2f} "
                        f"SL=${d['stop_loss']:<10.2f} TP=${d['take_profit']:<10.2f} "
                        f"size={d['size']:.6f} | {d['module']}"
                    )
                    lines.append(f"         Reasoning: {d.get('reasoning', '')}")
                else:
                    lines.append(
                        f"  {d.get('trade_id','')} | {d['date']} | CLOSE {d['side']:4s} "
                        f"{d['symbol']:12s} @ ${d['exit_price']:<10.2f} "
                        f"PnL=${d.get('pnl_usd', 0):+.4f} | {d['exit_reason']} | "
                        f"equity=${d['equity_after']:.2f}"
                    )
                lines.append("")

        # Rejected signals summary
        if rejected:
            lines.append("-" * 70)
            lines.append(f"  REJECTED SIGNALS ({len(rejected)} total)")
            lines.append("-" * 70)
            reject_reasons = {}
            for r in rejected:
                reason = r["data"]["reason"].split(":")[0].replace("VETO", "").strip()
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count}")
            lines.append("")

        # Equity curve
        lines.append("-" * 70)
        lines.append("  EQUITY CURVE (daily)")
        lines.append("-" * 70)
        for ec in self.equity_curve[::5]:  # Every 5th day
            bar_len = int((ec["equity"] - 90) * 2) if ec["equity"] > 90 else 0
            bar = "#" * max(0, min(40, bar_len))
            lines.append(f"  {ec['date'][:10]}  ${ec['equity']:>8.2f}  DD={ec['drawdown_pct']:>5.2f}%  {bar}")
        # Always show last day
        if self.equity_curve:
            ec = self.equity_curve[-1]
            bar_len = int((ec["equity"] - 90) * 2) if ec["equity"] > 90 else 0
            bar = "#" * max(0, min(40, bar_len))
            lines.append(f"  {ec['date'][:10]}  ${ec['equity']:>8.2f}  DD={ec['drawdown_pct']:>5.2f}%  {bar} <-- FINAL")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  VERIFICATION NOTICE")
        lines.append("  All prices sourced from Yahoo Finance (yfinance) and NSE (jugaad-data)")
        lines.append("  Timestamps correspond to actual trading sessions")
        lines.append("  Every trade can be verified against public market data")
        lines.append("  Commission: 0.1% applied both sides. Slippage: 0.05% per fill")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _log_event(self, event: str, data: dict):
        self.journal.append({
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        })

    def save_artifacts(self, report: str):
        """Save report + raw journal + equity curve."""
        out = Path("./reports/simulation")
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        (out / f"report_{ts}.txt").write_text(report, encoding="utf-8")
        (out / f"journal_{ts}.jsonl").write_text(
            "\n".join(json.dumps(j, default=str) for j in self.journal),
            encoding="utf-8",
        )
        (out / f"equity_{ts}.json").write_text(
            json.dumps(self.equity_curve, indent=2, default=str),
            encoding="utf-8",
        )
        return str(out)


async def main():
    sim = BattlefieldSimulation()

    print("=" * 50)
    print("  BATTLEFIELD SIMULATION")
    print(f"  Capital: ${INITIAL_CAPITAL} | Target: ${TARGET}")
    print("=" * 50)
    print()
    print("Fetching real market data...")

    datasets = await sim.fetch_market_data()
    if not datasets:
        print("ERROR: No market data available")
        return

    print(f"Loaded {len(datasets)} assets. Running simulation...")
    sim.run_simulation(datasets)

    report = sim.generate_report()
    print(report)

    path = sim.save_artifacts(report)
    print(f"\nArtifacts saved to: {path}")

    await sim.crypto.close()


if __name__ == "__main__":
    asyncio.run(main())
