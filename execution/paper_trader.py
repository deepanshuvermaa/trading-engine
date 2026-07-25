"""Paper trader — simulated live execution with realistic fills."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.loader import RiskConfig
from data.models import Signal, TradeSignal
from risk.engine import RiskEngine
from utils.logger import get_logger

log = get_logger("execution.paper")


class PaperTrader:
    """Paper trading engine for live simulation without real capital."""

    def __init__(
        self,
        risk_config: RiskConfig,
        initial_capital: float = 100_000.0,
        log_dir: str = "./data/storage/paper_trades",
    ):
        self.risk = RiskEngine(risk_config)
        self.risk.state.equity = initial_capital
        self.risk.state.peak_equity = initial_capital
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log: list[dict] = []

    def process_signal(self, signal: TradeSignal) -> dict:
        """Process a trade signal through risk management."""
        approved, reason, size = self.risk.approve_signal(signal)

        result = {
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "signal": signal.signal.value,
            "approved": approved,
            "reason": reason,
            "size": size,
            "entry_price": signal.entry_price,
            "module": signal.module_name,
            "confidence": signal.confidence,
            "equity": self.risk.state.equity,
            "drawdown_pct": self.risk.state.current_drawdown_pct,
        }

        if approved:
            self.risk.open_position(signal, size)
            log.info(
                f"PAPER OPEN: {signal.signal.value} {signal.symbol} @ {signal.entry_price:.2f} "
                f"size={size:.4f} | {signal.module_name}"
            )
        else:
            log.info(f"PAPER REJECTED: {signal.symbol} — {reason}")

        self.trade_log.append(result)
        return result

    def update_prices(self, prices: dict[str, tuple[float, float, float]]) -> list[dict]:
        """Update with current prices {symbol: (high, low, close)}. Check stops."""
        closed = []
        for symbol, (high, low, close) in prices.items():
            stop_hit = self.risk.check_stops(symbol, high, low)
            if stop_hit:
                pos = self.risk.state.open_positions[symbol]
                exit_price = pos.stop_loss if stop_hit == "sl" else pos.take_profit
                pnl = self.risk.close_position(symbol, exit_price)
                result = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "action": "CLOSE",
                    "exit_reason": stop_hit,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "equity": self.risk.state.equity,
                    "module": pos.module_name,
                }
                closed.append(result)
                self.trade_log.append(result)
                log.info(
                    f"PAPER CLOSE: {symbol} @ {exit_price:.2f} ({stop_hit}) "
                    f"PnL={pnl:.2f} | equity={self.risk.state.equity:.2f}"
                )
        return closed

    def get_state(self) -> dict:
        return {
            "equity": self.risk.state.equity,
            "peak_equity": self.risk.state.peak_equity,
            "drawdown_pct": self.risk.state.current_drawdown_pct,
            "open_positions": len(self.risk.state.open_positions),
            "exposure_pct": self.risk.state.total_exposure_pct,
            "is_scaled_down": self.risk.state.is_scaled_down,
        }

    def save_log(self) -> str:
        path = self.log_dir / f"paper_log_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(path, "a") as f:
            for entry in self.trade_log:
                f.write(json.dumps(entry, default=str) + "\n")
        count = len(self.trade_log)
        self.trade_log.clear()
        return f"Saved {count} entries to {path}"
