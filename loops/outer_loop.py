"""Karpathy outer loop — the bilevel meta-optimizer.

Watches the inner loop's experiment log. Analyzes which types of experiments
succeed and which modules are stagnating. Adjusts the inner loop's search bias.

This is the "autoresearch autoresearch" layer from the bilevel paper.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from config.loader import Settings
from knowledge.rules import RuleStats
from loops.experiment import ExperimentLog
from strategy.base import StrategyModule
from utils.logger import get_logger

log = get_logger("loop.outer")


class OuterLoop:
    """Meta-optimizer that improves the inner loop's search strategy."""

    def __init__(
        self,
        settings: Settings,
        experiment_log: ExperimentLog,
        rule_stats: RuleStats | None = None,
    ):
        self.settings = settings
        self.exp_log = experiment_log
        self.config = settings.loops.outer
        self.rule_stats = rule_stats or RuleStats()

    def analyze_and_adjust(
        self,
        modules: list[StrategyModule],
        inner_loop,
    ) -> dict:
        """Run the daily outer loop analysis.

        Returns a report dict with findings and actions taken.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "module_assessments": [],
            "search_bias_update": {},
            "rule_adjustments": [],
            "actions": [],
        }

        # 1. Analyze experiment success rates by parameter type
        type_rates = self.exp_log.success_rate_by_type()
        log.info(f"Experiment success rates by type: {type_rates}")

        # Compute new search bias — weight toward successful experiment types
        new_bias = {}
        if type_rates:
            max_rate = max(type_rates.values()) if type_rates else 1.0
            for param_type, rate in type_rates.items():
                if rate < self.config.experiment_success_rate_min:
                    # This param type is not worth exploring — suppress it
                    new_bias[param_type] = 0.1
                    report["actions"].append(
                        f"SUPPRESS {param_type}: success rate {rate:.1%} below threshold"
                    )
                else:
                    # Scale bias proportional to success rate
                    new_bias[param_type] = max(0.1, rate / max_rate)

            inner_loop.param_bias = new_bias
            report["search_bias_update"] = new_bias

        # 1b. Rule-performance feedback (loops/attribution.py -> knowledge/rule_stats.json).
        # Down-weight LEARNABLE investor rules that keep being wrong so the inner
        # loop stops exploring around them. RuleStats.underperformers() excludes
        # IMMUTABLE risk vetoes by construction — those may NEVER be relaxed.
        self.rule_stats._load()  # pick up updates written by the attribution loop
        weak_rules = self.rule_stats.underperformers(accuracy_below=0.4)
        rule_bias: dict[str, float] = {}
        for w in weak_rules:
            rule_bias[w["rule_id"]] = max(0.1, w["accuracy"])
            report["actions"].append(
                f"DOWN-WEIGHT rule {w['rule_id']} ({w['investor']}): "
                f"accuracy {w['accuracy']:.0%} over {w['samples']} trades -> weight {w['weight']}"
            )
            # Suppress inner-loop search around this rule's parameter family too
            bias = getattr(inner_loop, "param_bias", None)
            if isinstance(bias, dict):
                bias[w["category"]] = min(bias.get(w["category"], 1.0), 0.2)
        inner_loop.rule_bias = rule_bias
        report["rule_adjustments"] = weak_rules

        # 2. Assess each module's trajectory
        for module in modules:
            assessment = self._assess_module(module)
            report["module_assessments"].append(assessment)

            # 3. Trigger lifecycle transitions
            old_status = module.kpi.status
            new_status = module.evaluate_lifecycle(self.settings.modules)
            if old_status != new_status:
                report["actions"].append(
                    f"LIFECYCLE: {module.name} {old_status.value} → {new_status.value}"
                )
                log.info(f"Module {module.name}: {old_status.value} → {new_status.value}")

        return report

    def _assess_module(self, module: StrategyModule) -> dict:
        """Assess a single module's performance trajectory."""
        recent = self.exp_log.get_for_module(module.name)
        last_30 = [e for e in recent if e.timestamp > datetime.utcnow() - timedelta(days=30)]

        total = len(last_30)
        accepted = sum(1 for e in last_30 if e.accepted)
        rate = accepted / total if total > 0 else 0

        # Trend: compare last 15 experiments to previous 15
        if len(last_30) >= 10:
            mid = len(last_30) // 2
            first_half_rate = sum(1 for e in last_30[:mid] if e.accepted) / mid if mid else 0
            second_half_rate = sum(1 for e in last_30[mid:] if e.accepted) / (len(last_30) - mid) if (len(last_30) - mid) else 0
            if second_half_rate > first_half_rate + 0.05:
                trend = "IMPROVING"
            elif second_half_rate < first_half_rate - 0.05:
                trend = "DEGRADING"
            else:
                trend = "STABLE"
        else:
            trend = "INSUFFICIENT_DATA"

        # Stagnation check
        is_stagnant = (
            module.kpi.days_since_last_improvement >= self.config.stagnation_threshold_days
        )

        return {
            "module": module.name,
            "status": module.kpi.status.value,
            "sharpe_30d": module.kpi.sharpe_30d,
            "win_rate": module.kpi.win_rate,
            "composite_score": module.kpi.composite_score,
            "experiments_30d": total,
            "accepted_30d": accepted,
            "success_rate_30d": round(rate, 3),
            "trend": trend,
            "stagnant": is_stagnant,
            "weight": module.kpi.weight,
        }
