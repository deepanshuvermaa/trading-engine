"""
LIVE BATTLEFIELD — July 10, 2026
================================
Scan ALL markets RIGHT NOW. Find setups. Execute at current prices.
$100 capital. No excuses. Find the edge.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from config.loader import load_settings
from data.ingestion.crypto import CryptoProvider
from data.ingestion.us_equity import USEquityProvider
from data.ingestion.indian_equity import IndianEquityProvider
from indicators.technical import compute_all, rsi, macd, bollinger_bands, atr, ema, adx, volume_profile
from indicators.structural import detect_order_blocks, detect_fair_value_gaps, market_structure_break, support_resistance_levels
from data.models import Signal


# ── CONFIG ──────────────────────────────────────────────────────────
CAPITAL = 100.0
# Aggressive but controlled: 10% per trade, max 3 concurrent
MAX_POSITION_PCT = 10.0
MAX_CONCURRENT = 3
COMMISSION_PCT = 0.1
SLIPPAGE_PCT = 0.05


class LiveScanner:
    """Scan every available market for actionable setups RIGHT NOW."""

    def __init__(self):
        self.settings = load_settings()
        self.crypto = CryptoProvider("binance", self.settings.engine.data_dir)
        self.us_eq = USEquityProvider(self.settings.engine.data_dir)
        self.indian_eq = IndianEquityProvider(self.settings.engine.data_dir)
        self.now = datetime.now(timezone.utc)
        self.today = self.now.strftime("%Y-%m-%d")
        self.equity = CAPITAL
        self.trades = []
        self.scan_results = []

    async def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch latest data for entire universe."""
        datasets = {}
        start = self.now - timedelta(days=200)  # Need 200 bars for SMA200

        # CRYPTO — 24/7, always tradeable
        crypto_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
                          "BNB/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
                          "DOT/USDT", "LINK/USDT", "MATIC/USDT", "UNI/USDT"]

        for symbol in crypto_symbols:
            try:
                df = await self.crypto.get_ohlcv(symbol, "1d", start, self.now)
                if len(df) > 50:
                    datasets[symbol] = df
            except:
                pass

        # US EQUITIES — check if market hours
        us_symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
                      "SPY", "QQQ", "AMD", "COIN", "MSTR", "PLTR", "ARM"]

        for symbol in us_symbols:
            try:
                df = await self.us_eq.get_ohlcv(symbol, "1d", start, self.now)
                if len(df) > 50:
                    datasets[symbol] = df
            except:
                pass

        # INDIAN EQUITIES
        indian_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                          "BHARTIARTL", "SBIN", "ITC", "LT", "TATAMOTORS"]

        for symbol in indian_symbols:
            try:
                df = await self.indian_eq.get_ohlcv(symbol, "1d", start, self.now)
                if len(df) > 50:
                    datasets[symbol] = df
            except:
                pass

        return datasets

    def scan_asset(self, symbol: str, df: pd.DataFrame) -> dict | None:
        """Deep scan a single asset for trade setup. Returns setup or None."""
        if len(df) < 50:
            return None

        enriched = compute_all(df)
        last = enriched.iloc[-1]
        prev = enriched.iloc[-2]

        if pd.isna(last.get("rsi")) or pd.isna(last.get("atr")):
            return None

        current_price = float(last["close"])
        current_rsi = float(last["rsi"])
        current_adx = float(last["adx"]) if not pd.isna(last.get("adx")) else 0
        current_macd_hist = float(last["macd_hist"]) if not pd.isna(last.get("macd_hist")) else 0
        prev_macd_hist = float(prev["macd_hist"]) if not pd.isna(prev.get("macd_hist")) else 0
        current_atr = float(last["atr"])
        current_vol_ratio = float(last["vol_ratio"]) if not pd.isna(last.get("vol_ratio")) else 1.0
        bb_upper = float(last["bb_upper"]) if not pd.isna(last.get("bb_upper")) else 0
        bb_lower = float(last["bb_lower"]) if not pd.isna(last.get("bb_lower")) else 0
        bb_mid = float(last["bb_mid"]) if not pd.isna(last.get("bb_mid")) else 0
        ema_fast = float(last["ema_fast"]) if not pd.isna(last.get("ema_fast")) else 0
        ema_slow = float(last["ema_slow"]) if not pd.isna(last.get("ema_slow")) else 0
        sma_50 = float(last["sma_50"]) if not pd.isna(last.get("sma_50")) else 0
        sma_200 = float(last["sma_200"]) if not pd.isna(last.get("sma_200")) else 0

        # Price position relative to structure
        above_sma200 = current_price > sma_200 if sma_200 > 0 else None
        above_sma50 = current_price > sma_50 if sma_50 > 0 else None
        golden_cross = sma_50 > sma_200 if sma_50 > 0 and sma_200 > 0 else None

        # Recent swing high/low
        support_levels, resistance_levels = support_resistance_levels(
            enriched["high"], enriched["low"], lookback=5, num_levels=3
        )

        # Market structure break
        msb = market_structure_break(enriched["high"], enriched["low"])
        latest_msb = int(msb.iloc[-1]) if not pd.isna(msb.iloc[-1]) else 0

        # Order blocks
        ob = detect_order_blocks(enriched)
        recent_bullish_ob = ob["bullish_ob"].iloc[-10:].sum() > 0
        recent_bearish_ob = ob["bearish_ob"].iloc[-10:].sum() > 0

        # FVG
        fvg = detect_fair_value_gaps(enriched)
        recent_bullish_fvg = fvg["bullish_fvg"].iloc[-5:].sum() > 0
        recent_bearish_fvg = fvg["bearish_fvg"].iloc[-5:].sum() > 0

        # ── SCORING ENGINE ──────────────────────────────────────────
        # Score from -100 (strong sell) to +100 (strong buy)
        score = 0
        reasons = []

        # 1. Trend alignment
        if above_sma200:
            score += 15
            reasons.append("Above SMA200 (bullish trend)")
        elif above_sma200 is False:
            score -= 15
            reasons.append("Below SMA200 (bearish trend)")

        if golden_cross:
            score += 10
            reasons.append("Golden cross (SMA50 > SMA200)")
        elif golden_cross is False:
            score -= 10
            reasons.append("Death cross (SMA50 < SMA200)")

        # 2. EMA momentum
        if ema_fast > ema_slow and prev["ema_fast"] <= prev["ema_slow"]:
            score += 20
            reasons.append("FRESH bullish EMA crossover")
        elif ema_fast < ema_slow and prev["ema_fast"] >= prev["ema_slow"]:
            score -= 20
            reasons.append("FRESH bearish EMA crossover")
        elif ema_fast > ema_slow:
            score += 5
            reasons.append("EMA bullish")
        elif ema_fast < ema_slow:
            score -= 5
            reasons.append("EMA bearish")

        # 3. RSI
        if current_rsi < 30:
            score += 20
            reasons.append(f"RSI={current_rsi:.1f} OVERSOLD")
        elif current_rsi < 40:
            score += 10
            reasons.append(f"RSI={current_rsi:.1f} approaching oversold")
        elif current_rsi > 70:
            score -= 20
            reasons.append(f"RSI={current_rsi:.1f} OVERBOUGHT")
        elif current_rsi > 60:
            score -= 10
            reasons.append(f"RSI={current_rsi:.1f} approaching overbought")

        # 4. MACD momentum shift
        if current_macd_hist > 0 and prev_macd_hist <= 0:
            score += 15
            reasons.append("MACD histogram turned positive")
        elif current_macd_hist < 0 and prev_macd_hist >= 0:
            score -= 15
            reasons.append("MACD histogram turned negative")

        # 5. Bollinger Band position
        if current_price <= bb_lower:
            score += 15
            reasons.append("Price at BB lower band (mean reversion)")
        elif current_price >= bb_upper:
            score -= 15
            reasons.append("Price at BB upper band (overextended)")

        # 6. Volume confirmation
        if current_vol_ratio > 1.5:
            if score > 0:
                score += 10
                reasons.append(f"Volume surge {current_vol_ratio:.1f}x confirms move")
            elif score < 0:
                score -= 10
                reasons.append(f"Volume surge {current_vol_ratio:.1f}x confirms selling")

        # 7. ADX trend strength
        if current_adx > 25:
            reasons.append(f"ADX={current_adx:.1f} strong trend")
            score = int(score * 1.2)  # Amplify signal in trending market
        else:
            reasons.append(f"ADX={current_adx:.1f} ranging market")

        # 8. Structure
        if latest_msb == 1:
            score += 15
            reasons.append("Bullish market structure break")
        elif latest_msb == -1:
            score -= 15
            reasons.append("Bearish market structure break")

        if recent_bullish_ob and score > 0:
            score += 5
            reasons.append("Near bullish order block")
        if recent_bearish_ob and score < 0:
            score -= 5
            reasons.append("Near bearish order block")

        # 9. Multi-day momentum
        pct_5d = (current_price - float(enriched["close"].iloc[-6])) / float(enriched["close"].iloc[-6]) * 100
        pct_20d = (current_price - float(enriched["close"].iloc[-21])) / float(enriched["close"].iloc[-21]) * 100 if len(enriched) > 21 else 0

        # ── GENERATE SETUP ──────────────────────────────────────────
        if abs(score) < 25:
            return None  # No clear edge

        direction = Signal.BUY if score > 0 else Signal.SELL
        confidence = min(1.0, abs(score) / 100)

        # Risk/reward based on ATR
        atr_mult_sl = 2.0
        atr_mult_tp = 3.0

        if direction == Signal.BUY:
            stop_loss = current_price - atr_mult_sl * current_atr
            take_profit = current_price + atr_mult_tp * current_atr
        else:
            stop_loss = current_price + atr_mult_sl * current_atr
            take_profit = current_price - atr_mult_tp * current_atr

        risk_per_unit = abs(current_price - stop_loss)
        reward_per_unit = abs(take_profit - current_price)
        rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0

        return {
            "symbol": symbol,
            "direction": direction.value,
            "score": score,
            "confidence": round(confidence, 3),
            "current_price": round(current_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward": round(rr_ratio, 2),
            "atr": round(current_atr, 2),
            "rsi": round(current_rsi, 1),
            "adx": round(current_adx, 1),
            "macd_hist": round(current_macd_hist, 4),
            "volume_ratio": round(current_vol_ratio, 2),
            "pct_5d": round(pct_5d, 2),
            "pct_20d": round(pct_20d, 2),
            "above_sma200": above_sma200,
            "golden_cross": golden_cross,
            "reasons": reasons,
            "support": [round(s, 2) for s in support_levels[:3]],
            "resistance": [round(r, 2) for r in resistance_levels[:3]],
        }

    def rank_and_select(self, setups: list[dict]) -> list[dict]:
        """Rank setups by score * confidence * R:R. Select top trades."""
        for s in setups:
            s["rank_score"] = abs(s["score"]) * s["confidence"] * min(s["risk_reward"], 3.0)
        ranked = sorted(setups, key=lambda x: x["rank_score"], reverse=True)
        return ranked[:MAX_CONCURRENT]

    def execute_trades(self, selected: list[dict]):
        """Execute paper trades at current real prices."""
        for setup in selected:
            # Position sizing: fixed % of equity
            risk_amount = self.equity * (MAX_POSITION_PCT / 100)
            risk_per_unit = abs(setup["current_price"] - setup["stop_loss"])
            if risk_per_unit <= 0:
                continue

            size = risk_amount / risk_per_unit
            notional = size * setup["current_price"]

            # Cap notional to max position
            max_notional = self.equity * (MAX_POSITION_PCT / 100)
            if notional > max_notional:
                size = max_notional / setup["current_price"]
                notional = size * setup["current_price"]

            # Apply slippage + commission
            slip = setup["current_price"] * SLIPPAGE_PCT / 100
            entry = setup["current_price"] + slip if setup["direction"] == "BUY" else setup["current_price"] - slip
            commission = notional * COMMISSION_PCT / 100

            trade = {
                "trade_id": f"LIVE-{len(self.trades)+1:03d}",
                "timestamp": self.now.isoformat(),
                "date": self.today,
                "symbol": setup["symbol"],
                "direction": setup["direction"],
                "entry_price": round(entry, 2),
                "stop_loss": setup["stop_loss"],
                "take_profit": setup["take_profit"],
                "size": round(size, 6),
                "notional_usd": round(notional, 4),
                "commission": round(commission, 4),
                "risk_usd": round(risk_per_unit * size, 4),
                "potential_reward_usd": round(abs(setup["take_profit"] - entry) * size, 4),
                "risk_reward": setup["risk_reward"],
                "confidence": setup["confidence"],
                "score": setup["score"],
                "reasons": setup["reasons"],
                "status": "OPEN",
            }
            self.trades.append(trade)

    def generate_report(self, all_setups: list[dict], selected: list[dict]) -> str:
        """Generate comprehensive live trading report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  LIVE BATTLEFIELD REPORT — {self.today}")
        lines.append(f"  Generated: {self.now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Capital: ${CAPITAL:.2f} | Target: $120.00")
        lines.append(f"  Assets Scanned: {len(self.scan_results)}")
        lines.append(f"  Setups Found: {len(all_setups)}")
        lines.append(f"  Trades Taken: {len(selected)}")
        lines.append("")

        # ── MARKET OVERVIEW ──────────────────────────────────────────
        lines.append("-" * 70)
        lines.append("  MARKET SCAN RESULTS (all assets)")
        lines.append("-" * 70)
        lines.append(f"  {'Symbol':<14} {'Price':>10} {'RSI':>6} {'ADX':>6} {'5d%':>7} {'20d%':>7} {'Vol':>5} {'Verdict':<10}")
        lines.append(f"  {'------':<14} {'-----':>10} {'---':>6} {'---':>6} {'---':>7} {'----':>7} {'---':>5} {'-------':<10}")

        for sr in sorted(self.scan_results, key=lambda x: abs(x.get("score", 0)), reverse=True):
            verdict = "STRONG BUY" if sr.get("score", 0) > 50 else \
                      "BUY" if sr.get("score", 0) > 25 else \
                      "STRONG SELL" if sr.get("score", 0) < -50 else \
                      "SELL" if sr.get("score", 0) < -25 else \
                      "NEUTRAL"
            lines.append(
                f"  {sr['symbol']:<14} "
                f"${sr['current_price']:>9.2f} "
                f"{sr['rsi']:>5.1f} "
                f"{sr['adx']:>5.1f} "
                f"{sr['pct_5d']:>+6.1f}% "
                f"{sr['pct_20d']:>+6.1f}% "
                f"{sr['volume_ratio']:>4.1f}x "
                f"{verdict:<10}"
            )

        # ── REJECTED SETUPS ──────────────────────────────────────────
        not_selected = [s for s in all_setups if s not in selected]
        if not_selected:
            lines.append("")
            lines.append("-" * 70)
            lines.append("  SETUPS IDENTIFIED BUT NOT TAKEN")
            lines.append("-" * 70)
            for s in not_selected:
                lines.append(f"  {s['symbol']:<14} {s['direction']:>5} score={s['score']:>+4d} "
                             f"R:R={s['risk_reward']:.1f} conf={s['confidence']:.2f}")
                lines.append(f"    Reason not taken: Lower rank score vs selected trades")

        # ── EXECUTED TRADES ──────────────────────────────────────────
        lines.append("")
        lines.append("=" * 70)
        lines.append("  TRADES EXECUTED")
        lines.append("=" * 70)

        for t in self.trades:
            lines.append("")
            lines.append(f"  {t['trade_id']}")
            lines.append(f"  Date/Time:     {t['timestamp']}")
            lines.append(f"  Symbol:        {t['symbol']}")
            lines.append(f"  Direction:     {t['direction']}")
            lines.append(f"  Entry Price:   ${t['entry_price']}")
            lines.append(f"  Stop Loss:     ${t['stop_loss']}")
            lines.append(f"  Take Profit:   ${t['take_profit']}")
            lines.append(f"  Size:          {t['size']}")
            lines.append(f"  Notional:      ${t['notional_usd']}")
            lines.append(f"  Commission:    ${t['commission']}")
            lines.append(f"  Risk:          ${t['risk_usd']}")
            lines.append(f"  Potential:     ${t['potential_reward_usd']}")
            lines.append(f"  Risk/Reward:   1:{t['risk_reward']}")
            lines.append(f"  Confidence:    {t['confidence']*100:.0f}%")
            lines.append(f"  Score:         {t['score']}")
            lines.append(f"  Analysis:")
            for r in t["reasons"]:
                lines.append(f"    - {r}")

        # ── PORTFOLIO STATE ──────────────────────────────────────────
        lines.append("")
        lines.append("-" * 70)
        lines.append("  PORTFOLIO STATE AFTER EXECUTION")
        lines.append("-" * 70)
        total_risk = sum(t["risk_usd"] for t in self.trades)
        total_potential = sum(t["potential_reward_usd"] for t in self.trades)
        total_commission = sum(t["commission"] for t in self.trades)
        total_notional = sum(t["notional_usd"] for t in self.trades)
        exposure_pct = total_notional / CAPITAL * 100

        lines.append(f"  Equity:             ${CAPITAL:.2f}")
        lines.append(f"  Open Positions:     {len(self.trades)}")
        lines.append(f"  Total Notional:     ${total_notional:.2f}")
        lines.append(f"  Exposure:           {exposure_pct:.1f}%")
        lines.append(f"  Total Risk:         ${total_risk:.4f}")
        lines.append(f"  Total Potential:    ${total_potential:.4f}")
        lines.append(f"  Commission Paid:    ${total_commission:.4f}")
        lines.append(f"  Portfolio R:R:      1:{total_potential/total_risk:.1f}" if total_risk > 0 else "  Portfolio R:R:      N/A")

        # ── WHAT HAPPENS NEXT ────────────────────────────────────────
        lines.append("")
        lines.append("-" * 70)
        lines.append("  EXIT PLAN")
        lines.append("-" * 70)
        for t in self.trades:
            pct_to_sl = abs(t["entry_price"] - t["stop_loss"]) / t["entry_price"] * 100
            pct_to_tp = abs(t["take_profit"] - t["entry_price"]) / t["entry_price"] * 100
            lines.append(f"  {t['symbol']}")
            lines.append(f"    STOP LOSS at ${t['stop_loss']} ({pct_to_sl:.1f}% from entry) -> lose ${t['risk_usd']:.4f}")
            lines.append(f"    TAKE PROFIT at ${t['take_profit']} ({pct_to_tp:.1f}% from entry) -> gain ${t['potential_reward_usd']:.4f}")
            lines.append(f"    BREAK EVEN at ${t['entry_price'] + t['commission']/t['size']:.2f}" if t["size"] > 0 else "")

        # ── SCENARIO ANALYSIS ────────────────────────────────────────
        lines.append("")
        lines.append("-" * 70)
        lines.append("  SCENARIO ANALYSIS")
        lines.append("-" * 70)
        best = CAPITAL - total_commission + total_potential
        worst = CAPITAL - total_commission - total_risk
        lines.append(f"  ALL WIN:      ${best:.2f} ({(best-CAPITAL)/CAPITAL*100:+.2f}%)")
        lines.append(f"  ALL LOSE:     ${worst:.2f} ({(worst-CAPITAL)/CAPITAL*100:+.2f}%)")
        lines.append(f"  TARGET $120:  Need ${120-CAPITAL:.2f} more = "
                     f"{'POSSIBLE in one cycle' if total_potential >= 20 else f'{int(20/total_potential)+1} winning cycles needed' if total_potential > 0 else 'N/A'}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  DATA SOURCES")
        lines.append(f"  Crypto: Yahoo Finance (yfinance) via CCXT fallback")
        lines.append(f"  US Equities: Yahoo Finance (yfinance)")
        lines.append(f"  Indian Equities: NSE official data (jugaad-data)")
        lines.append(f"  All prices as of market close {self.today} or latest available")
        lines.append(f"  Indicators: TA-Lib (RSI, MACD, BB, ADX, ATR, EMA, SMA)")
        lines.append("=" * 70)

        return "\n".join(lines)

    async def run(self):
        print("=" * 50)
        print(f"  LIVE BATTLEFIELD — {self.today}")
        print(f"  Scanning all markets...")
        print("=" * 50)

        # 1. Fetch everything
        datasets = await self.fetch_all()
        print(f"  Loaded {len(datasets)} assets")

        # 2. Scan every asset
        all_setups = []
        for symbol, df in datasets.items():
            setup = self.scan_asset(symbol, df)
            if setup:
                self.scan_results.append(setup)
                if abs(setup["score"]) >= 25:
                    all_setups.append(setup)

        print(f"  Found {len(all_setups)} actionable setups from {len(self.scan_results)} scanned")

        # 3. Rank and select best
        selected = self.rank_and_select(all_setups)
        print(f"  Selected {len(selected)} trades to execute")

        # 4. Execute
        self.execute_trades(selected)

        # 5. Report
        report = self.generate_report(all_setups, selected)
        print(report)

        # 6. Save
        out = Path("./reports/live")
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out / f"live_report_{ts}.txt").write_text(report, encoding="utf-8")
        (out / f"trades_{ts}.json").write_text(
            json.dumps(self.trades, indent=2, default=str), encoding="utf-8"
        )
        (out / f"scan_{ts}.json").write_text(
            json.dumps(self.scan_results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n  Saved to: {out}")

        await self.crypto.close()


if __name__ == "__main__":
    asyncio.run(LiveScanner().run())
