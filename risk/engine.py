"""Risk engine — absolute veto power over all trading decisions.

This is the prepare.py equivalent. No strategy module can modify or bypass this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from config.loader import RiskConfig
from data.models import Signal, TradeSignal
from utils.logger import get_logger

log = get_logger("risk")


@dataclass
class PortfolioState:
    """Current portfolio risk state."""
    equity: float = 100_000.0  # Starting capital
    peak_equity: float = 100_000.0
    current_drawdown_pct: float = 0.0
    open_positions: dict[str, "Position"] = field(default_factory=dict)
    total_exposure_pct: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    is_scaled_down: bool = False


@dataclass
class Position:
    symbol: str
    side: Signal
    entry_price: float
    size: float  # in units
    stop_loss: float
    take_profit: float
    opened_at: datetime
    module_name: str

    @property
    def notional(self) -> float:
        return abs(self.size * self.entry_price)


class RiskEngine:
    """The compliance department. Has absolute veto power."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = PortfolioState()

    def approve_signal(self, signal: TradeSignal) -> tuple[bool, str, float]:
        """
        Evaluate a trade signal against risk rules.
        Returns: (approved, reason, position_size)
        """
        # Rule 1: Check drawdown limit
        if self.state.current_drawdown_pct >= self.config.max_drawdown_pct:
            return False, f"VETO: drawdown {self.state.current_drawdown_pct:.2f}% >= limit {self.config.max_drawdown_pct}%", 0.0

        # Rule 2: Check if already in position for this symbol
        if signal.symbol in self.state.open_positions:
            return False, f"VETO: already in position for {signal.symbol}", 0.0

        # Rule 3: Check total portfolio exposure
        if self.state.total_exposure_pct >= self.config.max_portfolio_exposure_pct:
            return False, f"VETO: portfolio exposure {self.state.total_exposure_pct:.1f}% >= limit {self.config.max_portfolio_exposure_pct}%", 0.0

        # Rule 4: Validate stop loss exists and is reasonable
        if signal.signal == Signal.BUY:
            risk_per_unit = signal.entry_price - signal.stop_loss
        else:
            risk_per_unit = signal.stop_loss - signal.entry_price

        if risk_per_unit <= 0:
            return False, "VETO: invalid stop loss — no risk defined", 0.0

        # Rule 5: Calculate position size (risk-based)
        max_risk_amount = self.state.equity * (self.config.max_position_pct / 100)

        # Scale down if in drawdown recovery
        if self.state.is_scaled_down:
            max_risk_amount *= self.config.scale_down_factor

        position_size = max_risk_amount / risk_per_unit

        # Rule 6: Ensure position doesn't exceed max position size
        notional = position_size * signal.entry_price
        max_notional = self.state.equity * (self.config.max_position_pct / 100)
        if notional > max_notional:
            position_size = max_notional / signal.entry_price

        # Rule 7: Confidence filter — reject low-confidence signals
        if signal.confidence < 0.3:
            return False, f"VETO: confidence {signal.confidence:.2f} below minimum 0.30", 0.0

        return True, "APPROVED", position_size

    def open_position(self, signal: TradeSignal, size: float) -> Position:
        """Record a new position."""
        pos = Position(
            symbol=signal.symbol,
            side=signal.signal,
            entry_price=signal.entry_price,
            size=size if signal.signal == Signal.BUY else -size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=signal.timestamp,
            module_name=signal.module_name,
        )
        self.state.open_positions[signal.symbol] = pos
        self._update_exposure()
        return pos

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return PnL."""
        if symbol not in self.state.open_positions:
            return 0.0

        pos = self.state.open_positions.pop(symbol)
        if pos.side == Signal.BUY:
            pnl = (exit_price - pos.entry_price) * abs(pos.size)
        else:
            pnl = (pos.entry_price - exit_price) * abs(pos.size)

        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.current_drawdown_pct = (
            (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
        )

        # Track win/loss streaks
        if pnl > 0:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
            if (self.state.is_scaled_down and
                    self.state.consecutive_wins >= self.config.recovery_streak_required):
                self.state.is_scaled_down = False
                log.info("Risk: recovery streak achieved, scaling back up")
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Auto scale-down on excessive drawdown
        if self.state.current_drawdown_pct >= self.config.max_drawdown_pct:
            self.state.is_scaled_down = True
            log.warning(f"Risk: drawdown {self.state.current_drawdown_pct:.2f}% — scaling down positions")

        self._update_exposure()
        return pnl

    def check_stops(self, symbol: str, high: float, low: float) -> str | None:
        """Check if stop loss or take profit hit. Returns 'sl', 'tp', or None."""
        if symbol not in self.state.open_positions:
            return None

        pos = self.state.open_positions[symbol]

        if pos.side == Signal.BUY:
            if low <= pos.stop_loss:
                return "sl"
            if high >= pos.take_profit:
                return "tp"
        else:
            if high >= pos.stop_loss:
                return "sl"
            if low <= pos.take_profit:
                return "tp"

        return None

    def _update_exposure(self):
        total_notional = sum(abs(p.notional) for p in self.state.open_positions.values())
        self.state.total_exposure_pct = (total_notional / self.state.equity * 100) if self.state.equity > 0 else 0
