"""KRA daily report generator — the module scorecard."""

from __future__ import annotations

from datetime import datetime

from loops.experiment import ExperimentLog
from strategy.base import StrategyModule


def generate_daily_report(
    modules: list[StrategyModule],
    experiment_log: ExperimentLog,
    outer_loop_report: dict | None = None,
) -> str:
    """Generate a text-based daily KRA report."""
    now = datetime.utcnow()
    lines = [
        "",
        "═" * 60,
        f"  DAILY MODULE PERFORMANCE REPORT — {now.strftime('%Y-%m-%d')}",
        "═" * 60,
    ]

    for module in modules:
        kpi = module.kpi
        exps = experiment_log.get_for_module(module.name)
        recent = exps[-50:] if exps else []
        accepted = sum(1 for e in recent if e.accepted)
        rejected = len(recent) - accepted

        # Trend arrow
        if kpi.consecutive_accepted >= 3:
            trend_icon = "▲"
            trend_text = f"IMPROVING — {kpi.consecutive_accepted} consecutive accepted"
        elif kpi.consecutive_rejected >= 5:
            trend_icon = "▼"
            trend_text = f"DEGRADING — {kpi.consecutive_rejected} consecutive rejected"
        else:
            trend_icon = "─"
            trend_text = "STABLE"

        # Action
        if kpi.status.value == "PROBATION":
            action = f"WEIGHT REDUCED TO {kpi.weight*100:.0f}% — recovery needed within {7 - kpi.days_since_last_improvement}d"
        elif kpi.status.value == "DEACTIVATED":
            action = "PARAMETERS RESET — re-entering trial"
        elif kpi.status.value == "TERMINATED":
            action = "MODULE REMOVED from registry"
        elif kpi.status.value == "TRIAL":
            days_left = 30 - (kpi.days_since_last_improvement if kpi.trial_start is None
                              else (now - kpi.trial_start).days)
            action = f"EVALUATION — {max(0, days_left)} days remaining in trial"
        else:
            action = "NONE — performing within threshold"

        lines.extend([
            "",
            f"MODULE: {module.name}",
            f"  Status:        {kpi.status.value}",
            f"  Sharpe (30d):  {kpi.sharpe_30d:.2f}  {trend_icon}",
            f"  Win Rate:      {kpi.win_rate*100:.1f}%",
            f"  Max Drawdown:  -{abs(kpi.max_drawdown_pct):.1f}%"
            + ("  ⚠ EXCEEDS LIMIT" if abs(kpi.max_drawdown_pct) > 2.0 else ""),
            f"  Profit Factor: {kpi.profit_factor:.2f}",
            f"  Composite:     {kpi.composite_score:.1f}/10",
            f"  Weight:        {kpi.weight*100:.0f}%",
            f"  Experiments:   {len(recent)} run, {accepted} accepted, {rejected} rejected",
            f"  Trend:         {trend_text}",
            f"  Action:        {action}",
        ])

    # Portfolio summary
    total_weight = sum(m.kpi.weight for m in modules)
    avg_score = (
        sum(m.kpi.composite_score * m.kpi.weight for m in modules) / total_weight
        if total_weight > 0 else 0
    )

    lines.extend([
        "",
        "─" * 60,
        f"PORTFOLIO COMPOSITE SCORE:  {avg_score:.1f} / 10",
    ])

    # Outer loop notes
    if outer_loop_report:
        bias = outer_loop_report.get("search_bias_update", {})
        if bias:
            suppressed = [k for k, v in bias.items() if v <= 0.1]
            boosted = [k for k, v in bias.items() if v >= 0.7]
            if suppressed:
                lines.append(f"META-OPTIMIZER: Suppressing {', '.join(suppressed)} experiments (low success rate)")
            if boosted:
                lines.append(f"META-OPTIMIZER: Boosting {', '.join(boosted)} experiments (high success rate)")

        for action in outer_loop_report.get("actions", []):
            lines.append(f"  {action}")

    lines.append("═" * 60)
    lines.append("")

    return "\n".join(lines)


def save_report(report: str, path: str = "./reports/daily") -> str:
    """Save report to file and return the path."""
    from pathlib import Path
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = out_dir / filename
    with open(filepath, "w") as f:
        f.write(report)
    return str(filepath)
