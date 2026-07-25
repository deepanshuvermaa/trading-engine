"""Karpathy inner loop — propose change → backtest → keep/discard → repeat.

This is train.py. The agent modifies parameters here.
The backtest engine (prepare.py) evaluates. If score improves → commit. If not → rollback.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

from backtest.engine import BacktestEngine
from config.loader import Settings
from loops.experiment import Experiment, ExperimentLog, propose_param_change
from strategy.base import StrategyModule
from utils.logger import get_logger

log = get_logger("loop.inner")


class InnerLoop:
    """The Karpathy inner loop — automated parameter search."""

    def __init__(
        self,
        settings: Settings,
        backtest_engine: BacktestEngine,
        experiment_log: ExperimentLog,
    ):
        self.settings = settings
        self.bt = backtest_engine
        self.exp_log = experiment_log
        self.loop_config = settings.loops.inner
        self.param_bias: dict[str, float] = {}  # Set by outer loop
        self.rule_bias: dict[str, float] = {}  # Set by outer loop from rule_stats (LEARNABLE only)

    def run_cycle(
        self,
        module: StrategyModule,
        data: pd.DataFrame,
        symbol: str,
    ) -> list[Experiment]:
        """Run N experiments on a module. Returns list of experiments."""

        experiments = []

        # Baseline score
        baseline_result = self.bt.run(module, data, symbol)
        baseline_score = baseline_result.composite_score

        for i in range(self.loop_config.experiments_per_cycle):
            # 1. PROPOSE a parameter change
            changes, exp_type = propose_param_change(module, self.param_bias)
            if not changes:
                continue

            # 2. APPLY the change
            old_params = module.update_params(changes)

            # 3. BACKTEST with new params
            new_result = self.bt.run(module, data, symbol)
            new_score = new_result.composite_score

            # 4. EVALUATE — did it improve?
            improvement = new_score - baseline_score
            accepted = improvement >= self.loop_config.min_sharpe_improvement

            # 5. KEEP or ROLLBACK
            if accepted:
                log.info(
                    f"  ACCEPTED: {module.name} | {exp_type}={changes.get(exp_type)} | "
                    f"score {baseline_score:.2f} → {new_score:.2f} (+{improvement:.2f})"
                )
                baseline_score = new_score  # new baseline
            else:
                module.rollback_params()

            # 6. RECORD experiment
            exp = Experiment(
                id=str(uuid.uuid4())[:8],
                module_name=module.name,
                timestamp=datetime.utcnow(),
                param_changes=changes,
                old_params=old_params,
                new_params=dict(module.params),
                old_score=baseline_score if not accepted else baseline_score - improvement,
                new_score=new_score,
                accepted=accepted,
                result=new_result,
                experiment_type=exp_type,
            )
            self.exp_log.record(exp)
            experiments.append(exp)

            # Update module KPI
            module.record_experiment(accepted)
            if accepted:
                module.update_kpi(new_result)

        return experiments
