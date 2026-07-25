"""Strategy module base class — every strategy inherits this.

Each module is a self-contained signal generator with:
- Tunable parameters (the part the Karpathy loop can modify)
- A generate_signals() method (deterministic, no LLM)
- KPI tracking (Sharpe, win rate, drawdown — updated by the evaluator)
- Lifecycle state (TRIAL → ACTIVE → PROBATION → DEACTIVATED → TERMINATED)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from data.models import BacktestResult, ModuleStatus, Signal, TradeSignal


class ModuleKPI(BaseModel):
    """Rolling KPI tracker for a strategy module."""
    sharpe_30d: float = 0.0
    sortino_30d: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_return_pct: float = 0.0
    composite_score: float = 0.0
    last_updated: datetime | None = None

    # Experiment tracking
    experiments_run: int = 0
    experiments_accepted: int = 0
    experiments_rejected: int = 0
    consecutive_accepted: int = 0
    consecutive_rejected: int = 0
    days_since_last_improvement: int = 0

    # Lifecycle
    status: ModuleStatus = ModuleStatus.TRIAL
    trial_start: datetime | None = None
    trial_failures: int = 0
    probation_start: datetime | None = None
    weight: float = 0.0  # signal weight in portfolio (0.0 to 1.0)


class StrategyModule(ABC):
    """Abstract base for all strategy modules.

    Subclasses must implement:
        - default_params() → dict of tunable parameters
        - generate_signals(df) → list of TradeSignals
    """

    name: str = "base"
    description: str = ""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.default_params(), **(params or {})}
        self.kpi = ModuleKPI(trial_start=datetime.utcnow())
        self._param_history: list[dict[str, Any]] = []

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Default tunable parameters. The Karpathy loop modifies these."""
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[TradeSignal]:
        """Generate trade signals from indicator-enriched DataFrame.

        The DataFrame already has all indicators computed (RSI, MACD, BBands, etc.)
        via indicators.technical.compute_all().
        """
        ...

    def get_param_ranges(self) -> dict[str, tuple[float, float, float]]:
        """Parameter ranges for the optimizer: {name: (min, max, step)}.
        Override in subclass for custom ranges.
        """
        return {}

    def update_params(self, new_params: dict[str, Any]) -> dict[str, Any]:
        """Update parameters and record history. Returns old params for rollback."""
        old = deepcopy(self.params)
        self._param_history.append(old)
        self.params.update(new_params)
        return old

    def rollback_params(self) -> bool:
        """Rollback to previous parameters."""
        if not self._param_history:
            return False
        self.params = self._param_history.pop()
        return True

    def update_kpi(self, result: BacktestResult) -> None:
        """Update KPI from a backtest result."""
        self.kpi.sharpe_30d = result.sharpe_ratio
        self.kpi.sortino_30d = result.sortino_ratio
        self.kpi.win_rate = result.win_rate
        self.kpi.max_drawdown_pct = result.max_drawdown_pct
        self.kpi.profit_factor = result.profit_factor
        self.kpi.total_trades = result.total_trades
        self.kpi.total_return_pct = result.total_return_pct
        self.kpi.composite_score = result.composite_score
        self.kpi.last_updated = datetime.utcnow()

    def record_experiment(self, accepted: bool) -> None:
        """Record an experiment result."""
        self.kpi.experiments_run += 1
        if accepted:
            self.kpi.experiments_accepted += 1
            self.kpi.consecutive_accepted += 1
            self.kpi.consecutive_rejected = 0
            self.kpi.days_since_last_improvement = 0
        else:
            self.kpi.experiments_rejected += 1
            self.kpi.consecutive_rejected += 1
            self.kpi.consecutive_accepted = 0

    def evaluate_lifecycle(self, config) -> ModuleStatus:
        """Evaluate module status based on KPI and lifecycle rules."""
        kpi = self.kpi
        now = datetime.utcnow()

        if kpi.status == ModuleStatus.TRIAL:
            if kpi.trial_start:
                days_in_trial = (now - kpi.trial_start).days
                if days_in_trial >= config.trial_period_days:
                    if (kpi.sharpe_30d >= config.trial_min_sharpe
                            and abs(kpi.max_drawdown_pct) <= config.trial_max_drawdown_pct):
                        kpi.status = ModuleStatus.ACTIVE
                        kpi.weight = min(1.0, kpi.sharpe_30d / 2.0)
                    else:
                        kpi.trial_failures += 1
                        if kpi.trial_failures >= config.max_trial_failures:
                            kpi.status = ModuleStatus.TERMINATED
                        else:
                            kpi.status = ModuleStatus.DEACTIVATED

        elif kpi.status == ModuleStatus.ACTIVE:
            if kpi.sharpe_30d < config.active_min_sharpe:
                kpi.status = ModuleStatus.PROBATION
                kpi.probation_start = now
                kpi.weight = 0.1
            else:
                kpi.weight = min(1.0, kpi.sharpe_30d / 2.0)

        elif kpi.status == ModuleStatus.PROBATION:
            if kpi.probation_start:
                days_on_probation = (now - kpi.probation_start).days
                if kpi.sharpe_30d >= config.active_min_sharpe:
                    kpi.status = ModuleStatus.ACTIVE
                    kpi.weight = min(1.0, kpi.sharpe_30d / 2.0)
                elif days_on_probation >= config.probation_days:
                    kpi.status = ModuleStatus.DEACTIVATED

        elif kpi.status == ModuleStatus.DEACTIVATED:
            # Reset params and re-enter trial
            self.params = self.default_params()
            self._param_history.clear()
            kpi.status = ModuleStatus.TRIAL
            kpi.trial_start = now

        return kpi.status

    def save_state(self, path: Path) -> None:
        """Persist module state to disk."""
        state = {
            "name": self.name,
            "params": self.params,
            "kpi": self.kpi.model_dump(mode="json"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: Path) -> bool:
        """Restore module state from disk."""
        if not path.exists():
            return False
        with open(path) as f:
            state = json.load(f)
        self.params = state.get("params", self.default_params())
        kpi_data = state.get("kpi", {})
        self.kpi = ModuleKPI(**kpi_data)
        return True
