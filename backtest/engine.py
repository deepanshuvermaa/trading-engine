"""Backtest engine — the verifier. No module can modify this.

Walk-forward backtesting with realistic transaction costs and slippage.
This is prepare.py — the immovable truth gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from config.loader import RiskConfig
from data.models import BacktestResult, Signal, TradeSignal
from indicators.technical import compute_all
from risk.engine import RiskEngine
from strategy.base import StrategyModule
from utils.logger import get_logger

log = get_logger("backtest")


@dataclass
class Trade:
    symbol: str
    side: Signal
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str  # "sl", "tp", "signal", "end"
    module_name: str


class BacktestEngine:
    """Walk-forward backtester. Immutable evaluation logic."""

    def __init__(
        self,
        risk_config: RiskConfig | None = None,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
        initial_capital: float = 100_000.0,
    ):
        self.risk_config = risk_config or RiskConfig()
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.initial_capital = initial_capital

    def run(
        self,
        module: StrategyModule,
        df: pd.DataFrame,
        symbol: str = "",
        indicator_params: dict | None = None,
    ) -> BacktestResult:
        """Run a full backtest for a module on given data."""

        if len(df) < 50:
            log.warning(f"Insufficient data for backtest: {len(df)} bars")
            return BacktestResult()

        # Compute indicators
        enriched = compute_all(df, indicator_params or module.params)
        enriched.attrs["symbol"] = symbol

        # Generate signals
        signals = module.generate_signals(enriched)
        if not signals:
            return BacktestResult()

        # Simulate execution
        trades = self._simulate(signals, enriched, symbol)
        if not trades:
            return BacktestResult()

        return self._compute_metrics(trades)

    def _simulate(
        self,
        signals: list[TradeSignal],
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Trade]:
        """Simulate trade execution with risk management."""
        risk = RiskEngine(self.risk_config)
        risk.state.equity = self.initial_capital
        risk.state.peak_equity = self.initial_capital

        trades: list[Trade] = []
        signal_idx = 0

        for i in range(len(df)):
            row = df.iloc[i]
            ts = df.index[i]

            # Check stops on open positions
            if symbol in risk.state.open_positions:
                stop_hit = risk.check_stops(symbol, row["high"], row["low"])
                if stop_hit:
                    pos = risk.state.open_positions[symbol]
                    exit_price = pos.stop_loss if stop_hit == "sl" else pos.take_profit
                    exit_price = self._apply_slippage(exit_price, pos.side == Signal.BUY)
                    pnl = risk.close_position(symbol, exit_price)
                    pnl -= abs(pos.notional) * self.commission_pct / 100  # Exit commission

                    trades.append(Trade(
                        symbol=symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        size=abs(pos.size),
                        entry_time=pos.opened_at,
                        exit_time=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        pnl=pnl,
                        pnl_pct=(pnl / pos.notional * 100) if pos.notional else 0,
                        exit_reason=stop_hit,
                        module_name=pos.module_name,
                    ))

            # Process signals at this timestamp
            while signal_idx < len(signals):
                sig = signals[signal_idx]
                sig_ts = pd.Timestamp(sig.timestamp)
                if sig_ts > ts:
                    break
                signal_idx += 1

                if sig_ts != ts:
                    continue
                if sig.signal == Signal.HOLD:
                    continue

                approved, reason, size = risk.approve_signal(sig)
                if not approved:
                    continue

                entry_price = self._apply_slippage(sig.entry_price, sig.signal == Signal.BUY)
                commission = abs(size * entry_price) * self.commission_pct / 100
                risk.state.equity -= commission

                risk.open_position(sig, size)

        # Close any remaining positions at last price
        last_row = df.iloc[-1]
        for sym in list(risk.state.open_positions.keys()):
            pos = risk.state.open_positions[sym]
            exit_price = last_row["close"]
            pnl = risk.close_position(sym, exit_price)
            trades.append(Trade(
                symbol=sym,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size=abs(pos.size),
                entry_time=pos.opened_at,
                exit_time=df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else df.index[-1],
                pnl=pnl,
                pnl_pct=(pnl / pos.notional * 100) if pos.notional else 0,
                exit_reason="end",
                module_name=pos.module_name,
            ))

        return trades

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        slip = price * self.slippage_pct / 100
        return price + slip if is_buy else price - slip

    def _compute_metrics(self, trades: list[Trade]) -> BacktestResult:
        """Compute performance metrics from trade list."""
        if not trades:
            return BacktestResult()

        pnls = [t.pnl for t in trades]
        returns_pct = [t.pnl_pct for t in trades]

        total_return = sum(pnls)
        total_return_pct = total_return / self.initial_capital * 100

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (annualized, assuming daily returns)
        returns_array = np.array(returns_pct)
        if len(returns_array) > 1 and returns_array.std() > 0:
            sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (only downside deviation)
        downside = returns_array[returns_array < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (returns_array.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max drawdown from equity curve
        equity_curve = [self.initial_capital]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak * 100
        max_dd = drawdown.max()

        # Calmar ratio
        calmar = (total_return_pct / max_dd) if max_dd > 0 else 0

        return BacktestResult(
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            total_trades=len(trades),
            total_return_pct=round(total_return_pct, 4),
            avg_trade_return_pct=round(np.mean(returns_pct), 4),
            calmar_ratio=round(calmar, 4),
        )

    def walk_forward(
        self,
        module: StrategyModule,
        df: pd.DataFrame,
        symbol: str,
        train_pct: float = 0.7,
        n_splits: int = 5,
    ) -> list[BacktestResult]:
        """Walk-forward optimization with out-of-sample validation."""
        total_len = len(df)
        split_size = total_len // n_splits
        results = []

        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size, total_len)
            split_data = df.iloc[start:end]

            train_end = int(len(split_data) * train_pct)
            test_data = split_data.iloc[train_end:]

            if len(test_data) < 30:
                continue

            result = self.run(module, test_data, symbol)
            results.append(result)

        return results


@dataclass
class WalkForwardScore:
    """The confirmed metric's output for one module across a set of assets.

    `vetoed=True` means the 2% max-drawdown gate was breached on the held-out
    data — treated as an automatic FAIL (None) by `evaluate()`.
    """
    holdout_sharpe: float
    holdout_max_dd_pct: float
    holdout_return_pct: float
    total_trades: int
    assets_scored: int
    vetoed: bool


class WalkForwardEvaluator:
    """The CONFIRMED metric — Sharpe on the held-out UNSEEN period, gated by the
    2% max-drawdown veto (see .loop/anchors/metrics.md + holdout.md).

    Holds out the chronologically LAST `holdout_pct` of each asset's history (the
    most-recent, never-tuned-on slice), runs the immutable BacktestEngine on it,
    aggregates Sharpe across assets, and FAILS (returns None) if the aggregate
    held-out max drawdown breaches `max_drawdown_pct`.
    """

    def __init__(
        self,
        engine: "BacktestEngine | None" = None,
        holdout_pct: float = 0.20,
        max_drawdown_pct: float = 2.0,
        min_bars: int = 60,
        min_holdout_bars: int = 30,
    ):
        self.engine = engine or BacktestEngine()
        self.holdout_pct = holdout_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_bars = min_bars
        self.min_holdout_bars = min_holdout_bars

    def _holdout(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or len(df) < self.min_bars:
            return None
        split = int(len(df) * (1.0 - self.holdout_pct))
        hold = df.iloc[split:]
        return hold if len(hold) >= self.min_holdout_bars else None

    def score(
        self,
        module: StrategyModule,
        datasets: dict[str, pd.DataFrame],
    ) -> WalkForwardScore | None:
        """Aggregate held-out score for one module. None if no scorable data."""
        results: list[BacktestResult] = []
        for symbol, df in datasets.items():
            hold = self._holdout(df)
            if hold is None:
                continue
            res = self.engine.run(module, hold, symbol)
            if res.total_trades > 0:
                results.append(res)

        if not results:
            return None

        sharpes = [r.sharpe_ratio for r in results]
        dds = [abs(r.max_drawdown_pct) for r in results]
        rets = [r.total_return_pct for r in results]
        trades = sum(r.total_trades for r in results)

        holdout_sharpe = float(np.mean(sharpes))
        holdout_max_dd = float(max(dds)) if dds else 0.0
        vetoed = holdout_max_dd > self.max_drawdown_pct
        return WalkForwardScore(
            holdout_sharpe=round(holdout_sharpe, 4),
            holdout_max_dd_pct=round(holdout_max_dd, 4),
            holdout_return_pct=round(float(np.mean(rets)), 4),
            total_trades=trades,
            assets_scored=len(results),
            vetoed=vetoed,
        )

    def evaluate(
        self,
        module: StrategyModule,
        datasets: dict[str, pd.DataFrame],
    ) -> WalkForwardScore | None:
        """The confirmed-metric gate: returns the score, or **None** if the 2%
        max-drawdown veto is breached on the holdout (or no scorable data)."""
        s = self.score(module, datasets)
        if s is None:
            return None
        if s.vetoed:
            log.warning(
                f"WalkForward VETO: {module.name} held-out max DD "
                f"{s.holdout_max_dd_pct:.2f}% > {self.max_drawdown_pct:.2f}% — "
                f"metric fails (None). (Sharpe would have been {s.holdout_sharpe:.3f})"
            )
            return None
        return s
