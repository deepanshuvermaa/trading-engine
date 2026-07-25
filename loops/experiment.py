"""Experiment definition — a single parameter change + backtest cycle."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from data.models import BacktestResult


class Experiment(BaseModel):
    """Record of a single experiment."""
    id: str
    module_name: str
    timestamp: datetime
    param_changes: dict[str, Any]  # what was changed
    old_params: dict[str, Any]
    new_params: dict[str, Any]
    old_score: float
    new_score: float
    accepted: bool
    result: BacktestResult
    experiment_type: str = ""  # e.g., "rsi_period", "ema_fast", etc.

    @property
    def improvement(self) -> float:
        return self.new_score - self.old_score


class ExperimentLog:
    """Persistent experiment history — the state file for the Karpathy loop."""

    def __init__(self, path: str = "./data/storage/experiments.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._experiments: list[Experiment] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._experiments.append(Experiment.model_validate_json(line))

    def record(self, exp: Experiment) -> None:
        self._experiments.append(exp)
        with open(self.path, "a") as f:
            f.write(exp.model_dump_json() + "\n")

    def get_all(self) -> list[Experiment]:
        return self._experiments

    def get_for_module(self, module_name: str) -> list[Experiment]:
        return [e for e in self._experiments if e.module_name == module_name]

    def success_rate(self, module_name: str | None = None) -> float:
        exps = self.get_for_module(module_name) if module_name else self._experiments
        if not exps:
            return 0.0
        return sum(1 for e in exps if e.accepted) / len(exps)

    def success_rate_by_type(self) -> dict[str, float]:
        """Which types of experiments succeed most?"""
        by_type: dict[str, list[bool]] = {}
        for e in self._experiments:
            t = e.experiment_type or "unknown"
            by_type.setdefault(t, []).append(e.accepted)
        return {t: sum(v) / len(v) for t, v in by_type.items() if v}

    def recent(self, n: int = 50) -> list[Experiment]:
        return self._experiments[-n:]


def propose_param_change(
    module,
    bias: dict[str, float] | None = None,
) -> tuple[dict[str, Any], str]:
    """Propose a random parameter change within the module's defined ranges.

    Args:
        module: StrategyModule instance
        bias: optional dict mapping param names to probability weights
              (from outer loop analysis of which params are worth exploring)

    Returns: (param_changes dict, experiment_type string)
    """
    ranges = module.get_param_ranges()
    if not ranges:
        return {}, "none"

    # Choose which parameter to modify — biased by outer loop weights
    param_names = list(ranges.keys())
    if bias:
        weights = [bias.get(p, 1.0) for p in param_names]
        total = sum(weights)
        weights = [w / total for w in weights]
        param_name = random.choices(param_names, weights=weights, k=1)[0]
    else:
        param_name = random.choice(param_names)

    min_val, max_val, step = ranges[param_name]
    current = module.params.get(param_name, min_val)

    # Propose a new value — prefer small steps near current value
    n_steps = int((max_val - min_val) / step)
    possible = [min_val + i * step for i in range(n_steps + 1)]
    possible = [v for v in possible if abs(v - current) > step * 0.01]  # exclude current

    if not possible:
        return {}, "none"

    # Gaussian bias toward small changes
    distances = [abs(v - current) for v in possible]
    max_dist = max(distances) if distances else 1
    weights = [max(0.01, 1.0 - d / max_dist) for d in distances]
    new_val = random.choices(possible, weights=weights, k=1)[0]

    # Round to step precision
    if isinstance(step, int) or step == int(step):
        new_val = int(round(new_val))
    else:
        new_val = round(new_val, 4)

    return {param_name: new_val}, param_name
