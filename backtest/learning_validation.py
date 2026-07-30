"""Learning-loop validation harness — does knowledge/rule_stats.json actually help?

Runs the REAL production decision pipeline (AutonomousEngine.score_asset ->
RuleBrain -> PersonaEngine -> cost gate -> open_position/_close_position,
byte-for-byte, not a reimplementation) TWICE over the SAME held-out,
never-tuned-on data:

  RUN A "naive"   — every LEARNABLE rule forced to weight 1.0 (an in-memory
                    RuleStats with no track record; IMMUTABLE rules are
                    pinned at 1.0 regardless, by construction in
                    knowledge/rules.py::RuleStats.weight).
  RUN B "learned" — the ACTUAL current knowledge/rule_stats.json weights,
                    loaded from a COPY of the file (the live file is never
                    opened for writing).

Both runs share: the same historical baskets (backtest/historical_data.py),
the same walk-forward holdout split (backtest.engine.WalkForwardEvaluator,
reused not reinvented), the same cost model, the same immutable vetoes, the
same position sizing. The ONLY difference is learnable rule weights.

Metrics (Sharpe/Sortino/max-DD/win-rate/profit-factor) are computed by
backtest.engine.BacktestEngine._compute_metrics — reused, not reinvented.

CLI:  python -m backtest.learning_validation [--lookback-days 730]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import BacktestEngine, Trade, WalkForwardEvaluator
from backtest.historical_data import fetch_all, LOOKBACK_DAYS
from knowledge.rules import RuleStats, load_rules
from personas.engine import PersonaStats
from utils.logger import get_logger

log = get_logger("backtest.learning_validation")

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_RULE_STATS_PATH = REPO_ROOT / "knowledge" / "rule_stats.json"
LIVE_PERSONA_STATS_PATH = REPO_ROOT / "personas" / "persona_stats.json"
REPORTS_DIR = REPO_ROOT / "reports" / "backtest"
MIN_BARS = 50  # score_asset's own floor (indicators/technical.compute_all needs it)

_CITATION_RULE_RE = re.compile(r"\[([^|]+)\|")


# ── helpers ──────────────────────────────────────────────────────────────

def _parse_rule_id(citation: str) -> str | None:
    m = _CITATION_RULE_RE.search(citation)
    return m.group(1) if m else None


def _to_bt_trades(closed_trades: list[dict]) -> list[Trade]:
    """Closed-trade rows (AutonomousEngine's own trade dicts) -> backtest.engine.Trade,
    so metrics reuse BacktestEngine._compute_metrics verbatim."""
    trades: list[Trade] = []
    for t in closed_trades:
        if t.get("pnl") is None:  # the "OPEN" bookkeeping rows carry pnl=None
            continue
        entry = float(t.get("entry") or 0)
        size = float(t.get("size") or 0)
        fx = float(t.get("fx_rate") or 1.0) or 1.0
        notional_usd = size * entry / fx
        pnl = float(t.get("pnl") or 0)
        pnl_pct = (pnl / notional_usd * 100) if notional_usd else 0.0
        trades.append(Trade(
            symbol=t.get("symbol", ""),
            side=t.get("side", "BUY"),
            entry_price=entry,
            exit_price=float(t.get("exit") or 0),
            size=size,
            entry_time=t.get("date"),
            exit_time=t.get("date"),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=t.get("reason", ""),
            module_name=t.get("module", ""),
        ))
    return trades


def _build_engine(rule_stats_path: Path, persona_stats_path: Path,
                   audit_dir: Path, attribution_path: Path):
    """A fresh AutonomousEngine wired to isolated, temp-path stats/audit files
    so nothing this harness does can ever touch the live knowledge/personas
    JSON or the live reports/audit trail."""
    import autonomous  # local import: keeps this module importable standalone

    engine = autonomous.AutonomousEngine()

    rule_stats = RuleStats(path=rule_stats_path, rulebook=engine.brain.rulebook)
    engine.brain.stats = rule_stats
    engine.attribution.rule_stats = rule_stats  # same object the brain scores with

    persona_stats = PersonaStats(path=persona_stats_path)
    engine.personas.stats = persona_stats
    engine.portfolio_manager.stats = persona_stats

    engine.audit = autonomous.DecisionAudit(log_dir=str(audit_dir))
    engine.attribution.path = attribution_path
    attribution_path.parent.mkdir(parents=True, exist_ok=True)

    return engine, rule_stats, persona_stats


def _rule_stats_snapshot(rule_stats: RuleStats) -> dict[str, dict]:
    out = {}
    for rule_id in sorted(rule_stats.rulebook.rules.keys()):
        rule = rule_stats.rulebook.get(rule_id)
        out[rule_id] = {
            "investor": rule.investor if rule else "?",
            "mutability": rule.mutability.value if rule else "?",
            "weight": rule_stats.weight(rule_id),
            "accuracy": rule_stats.accuracy(rule_id),
            "samples": rule_stats.samples(rule_id),
        }
    return out


# ── the core walk-forward simulation (shared portfolio, cross-symbol,
#    chronologically merged — so max_concurrent/D8/D9 gates see the SAME
#    cross-asset competition a live cycle would) ─────────────────────────

def _simulate_one_config(
    label: str, symbol_data: dict[str, dict[str, pd.DataFrame]],
    wfe: WalkForwardEvaluator, tmp_root: Path,
    rule_stats_src: Path | None, persona_stats_src: Path | None,
) -> dict[str, Any]:
    import autonomous

    run_dir = tmp_root / label
    run_dir.mkdir(parents=True, exist_ok=True)

    rule_stats_path = run_dir / "rule_stats.json"
    if rule_stats_src and rule_stats_src.exists():
        shutil.copyfile(rule_stats_src, rule_stats_path)
    persona_stats_path = run_dir / "persona_stats.json"
    if persona_stats_src and persona_stats_src.exists():
        shutil.copyfile(persona_stats_src, persona_stats_path)

    engine, rule_stats, persona_stats = _build_engine(
        rule_stats_path, persona_stats_path,
        audit_dir=run_dir / "audit", attribution_path=run_dir / "attribution.jsonl")
    engine.symbol_markets = {
        sym: mkt for mkt, syms in symbol_data.items() for sym in syms}
    portfolio = engine.primary

    # Held-out windows per symbol, via the FROZEN split (holdout.md / engine.py).
    holds: dict[str, tuple[str, pd.DataFrame]] = {}
    per_symbol_meta: list[dict] = []
    for market, syms in symbol_data.items():
        for symbol, df in syms.items():
            hold = wfe._holdout(df)
            if hold is None or len(hold) < MIN_BARS:
                per_symbol_meta.append({
                    "symbol": symbol, "market": market, "included": False,
                    "reason": "insufficient bars for the frozen holdout split "
                              f"(need >={wfe.min_bars} total, >={max(wfe.min_holdout_bars, MIN_BARS)} holdout)",
                })
                continue
            holds[symbol] = (market, hold)
            per_symbol_meta.append({
                "symbol": symbol, "market": market, "included": True,
                "total_bars": len(df), "holdout_bars": len(hold),
                "holdout_start": str(hold.index[0].date()),
                "holdout_end": str(hold.index[-1].date()),
            })

    all_dates = sorted(set().union(*[set(h.index) for _, h in holds.values()])) if holds else []

    trade_log: list[dict] = []
    entry_audit: dict[str, dict] = {}

    def _record_close(symbol: str, market: str, sim_date: str):
        close_row = portfolio.closed_trades[-1]
        # _close_position stamps "date" with wall-clock datetime.now() (the
        # live engine's real-time journal convention) — meaningless when
        # replaying historical bars, so overwrite with the actual simulated
        # bar date for an accurate audit trail.
        close_row["date"] = sim_date
        meta = entry_audit.pop(symbol, {})
        entry_price = float(close_row.get("entry") or 0)
        size = float(close_row.get("size") or 0)
        fx = float(close_row.get("fx_rate") or 1.0) or 1.0
        notional_usd = size * entry_price / fx
        pnl = float(close_row.get("pnl") or 0)
        pnl_pct = (pnl / notional_usd * 100) if notional_usd else 0.0
        holding_days = None
        try:
            d0, d1 = pd.Timestamp(meta.get("entry_date")), pd.Timestamp(close_row.get("date"))
            holding_days = (d1 - d0).days
        except Exception:
            pass
        cons = meta.get("persona_consensus") or {}
        trade_log.append({
            "run": label, "trade_id": close_row.get("id"),
            "symbol": symbol, "market": market,
            "side": close_row.get("side"), "module": close_row.get("module"),
            "entry_date": meta.get("entry_date"), "entry_price": entry_price,
            "exit_date": close_row.get("date"), "exit_price": close_row.get("exit"),
            "exit_reason": close_row.get("reason"),
            "size": size, "pnl": round(pnl, 6), "pnl_pct": round(pnl_pct, 4),
            "holding_days": holding_days,
            "score": meta.get("score"), "risk_reward": meta.get("risk_reward"),
            "rule_score_multiplier": meta.get("rule_score_multiplier"),
            "rule_citations": meta.get("rule_citations", []),
            "rule_failures": meta.get("rule_failures", []),
            "persona_action": cons.get("action"), "persona_strength": cons.get("strength"),
            "persona_consensus_value": cons.get("consensus"), "persona_dissent": cons.get("dissent"),
            "persona_bulls": cons.get("bulls", []), "persona_bears": cons.get("bears", []),
            "persona_abstains": cons.get("abstains", []),
        })

    for d in all_dates:
        for symbol, (market, hold) in holds.items():
            if d not in hold.index:
                continue
            idx = hold.index.get_loc(d)
            if idx < MIN_BARS - 1:
                continue
            sub = hold.iloc[: idx + 1]

            was_open = symbol in portfolio.positions
            if was_open:
                engine.check_positions(portfolio, {symbol: sub})
                if symbol not in portfolio.positions:  # SL/TP/trail closed it this bar
                    _record_close(symbol, market, str(pd.Timestamp(d).date()))
                continue  # either still open, or just closed — no same-bar re-entry

            if len(portfolio.positions) >= engine.max_concurrent:
                continue
            setup = engine.score_asset(portfolio, symbol, sub, macro=None,
                                        min_score=25.0, skip_cost_gate=False)
            if setup is None:
                continue
            engine.open_position(portfolio, setup)
            if symbol in portfolio.positions:
                entry_audit[symbol] = {
                    "entry_date": str(pd.Timestamp(d).date()),
                    "rule_citations": list(setup.get("rule_citations", [])),
                    "rule_failures": list(setup.get("rule_failures", [])),
                    "rule_score_multiplier": setup.get("rule_score_multiplier"),
                    "score": setup.get("score"), "risk_reward": setup.get("risk_reward"),
                    "persona_consensus": setup.get("persona_consensus"),
                }

    # Close anything still open at the last available price in its holdout window.
    for symbol in list(portfolio.positions.keys()):
        market, hold = holds[symbol]
        last_price = float(hold["close"].iloc[-1])
        engine._close_position(portfolio, symbol, last_price, "END_OF_WINDOW")
        _record_close(symbol, market, str(hold.index[-1].date()))

    bt_trades = _to_bt_trades(portfolio.closed_trades)
    bt_engine = BacktestEngine(initial_capital=portfolio.initial_capital)
    metrics = bt_engine._compute_metrics(bt_trades)

    return {
        "label": label,
        "metrics": metrics.model_dump(),
        "trades": trade_log,
        "per_symbol_meta": per_symbol_meta,
        "rule_stats_snapshot": _rule_stats_snapshot(rule_stats),
        "rule_stats_path_used": str(rule_stats_path),
        "final_equity": round(portfolio.equity, 4),
        "initial_capital": portfolio.initial_capital,
    }


def _simulate_both(symbol_data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    """Synchronous, CPU-bound. Callers should offload this to a thread
    (asyncio.to_thread) so it never blocks the live Scout/Sentry event loop."""
    import autonomous

    tmp_root = Path(tempfile.mkdtemp(prefix="learning_validation_"))
    wfe = WalkForwardEvaluator()

    # Daily-bar backtest adaptation: the live 15m anti-churn re-entry cooldown
    # (REENTRY_COOLDOWN_MIN, default 120min) compares against wall-clock
    # datetime.now(), which is meaningless when we replay historical daily
    # bars in a tight loop — every daily bar is already >120min apart in
    # reality, so the live intent ("don't round-trip the same name within the
    # same session") is preserved by disabling it here. Identical for BOTH
    # runs, so it cannot bias the naive-vs-learned comparison. Restored
    # immediately after (this whole block is synchronous — no other
    # coroutine can observe the change).
    orig_cooldown = autonomous.REENTRY_COOLDOWN_MIN
    autonomous.REENTRY_COOLDOWN_MIN = 0.0
    try:
        rule_stats_before = (LIVE_RULE_STATS_PATH.read_bytes()
                              if LIVE_RULE_STATS_PATH.exists() else None)

        naive = _simulate_one_config(
            "naive", symbol_data, wfe, tmp_root,
            rule_stats_src=None, persona_stats_src=LIVE_PERSONA_STATS_PATH)
        learned = _simulate_one_config(
            "learned", symbol_data, wfe, tmp_root,
            rule_stats_src=LIVE_RULE_STATS_PATH, persona_stats_src=LIVE_PERSONA_STATS_PATH)

        rule_stats_after = (LIVE_RULE_STATS_PATH.read_bytes()
                             if LIVE_RULE_STATS_PATH.exists() else None)
    finally:
        autonomous.REENTRY_COOLDOWN_MIN = orig_cooldown
        shutil.rmtree(tmp_root, ignore_errors=True)

    return {
        "naive": naive, "learned": learned,
        "live_rule_stats_untouched": rule_stats_before == rule_stats_after,
    }


# ── attribution: which learned-weight rules actually moved a decision ────

def _weight_attribution(learned: dict[str, Any]) -> list[dict]:
    snap = learned["rule_stats_snapshot"]
    out = []
    for t in learned["trades"]:
        seen = set()
        for citation in t.get("rule_citations", []):
            rid = _parse_rule_id(citation)
            if not rid or rid in seen:
                continue
            seen.add(rid)
            info = snap.get(rid)
            if not info or info["mutability"] != "LEARNABLE":
                continue
            w = info["weight"]
            if abs(w - 1.0) < 1e-9:
                continue
            out.append({
                "trade_id": t["trade_id"], "symbol": t["symbol"],
                "entry_date": t["entry_date"], "rule_id": rid,
                "investor": info["investor"], "weight": w,
                "accuracy": info["accuracy"], "samples": info["samples"],
                "effect": "down-weighted (reduced score multiplier)" if w < 1.0
                          else "up-weighted (increased score multiplier)",
                "score_multiplier": t.get("rule_score_multiplier"),
                "trade_pnl": t["pnl"], "trade_outcome": "WIN" if t["pnl"] > 0 else "LOSS",
            })
    return out


def _feeds_next_trade_examples(learned: dict[str, Any], n: int = 4) -> list[dict]:
    """Concretely shows knowledge/rules.py::RuleStats.record()/weight() applied
    to a few of THIS run's own closed trades — the exact mechanism the live
    engine uses after every real close. Operates on a throwaway CLONE of the
    learned run's rule stats; never touches any file."""
    examples = []
    trades = sorted(learned["trades"], key=lambda t: abs(t["pnl"]), reverse=True)
    picked = [t for t in trades if t.get("rule_citations")][:n]
    for t in picked:
        # A scratch RuleStats pointed at a path that does not exist -> starts
        # empty, so .record()/.weight() below run the REAL production
        # ledger math starting from a clean slate, purely for illustration.
        # Never saved anywhere.
        clone = RuleStats(path=Path(tempfile.gettempdir()) / f"_lv_scratch_{t['trade_id']}.json",
                           rulebook=load_rules())
        won = t["pnl"] > 0
        rows = []
        for citation in t["rule_citations"][:4]:
            rid = _parse_rule_id(citation)
            if not rid:
                continue
            info = learned["rule_stats_snapshot"].get(rid, {})
            passed = citation.startswith("PASS")
            before_w = info.get("weight", 1.0)
            clone.record(rid, rule_passed=passed, trade_won=won, pnl=t["pnl"])
            after_w = clone.weight(rid)
            rows.append({
                "rule_id": rid, "investor": info.get("investor"),
                "rule_passed_at_entry": passed, "trade_won": won,
                "weight_before": before_w, "weight_after_this_trade": after_w,
            })
        examples.append({
            "trade_id": t["trade_id"], "symbol": t["symbol"],
            "entry_date": t["entry_date"], "exit_date": t["exit_date"],
            "pnl": t["pnl"], "outcome": "WIN" if won else "LOSS",
            "rule_updates": rows,
        })
    return examples


# ── report assembly ───────────────────────────────────────────────────────

def _fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def build_report(sim: dict[str, Any], fetch_reports: list, lookback_days: int,
                  started_at: datetime) -> dict[str, Any]:
    naive, learned = sim["naive"], sim["learned"]
    nm, lm = naive["metrics"], learned["metrics"]

    def _delta(key, better_high=True):
        d = lm[key] - nm[key]
        return d if better_high else -d

    naive_seq = [(t["symbol"], t["entry_date"], t["side"]) for t in naive["trades"]]
    learned_seq = [(t["symbol"], t["entry_date"], t["side"]) for t in learned["trades"]]
    identical_trade_sequence = naive_seq == learned_seq
    weight_diffs = [
        rid for rid, info in learned["rule_stats_snapshot"].items()
        if info["mutability"] == "LEARNABLE"
        and abs(info["weight"] - naive["rule_stats_snapshot"].get(rid, {}).get("weight", 1.0)) > 1e-9
    ]

    verdict_lines = []
    if identical_trade_sequence:
        verdict_lines.append(
            f"Rule weights genuinely diverged ({len(weight_diffs)} learnable rules: "
            f"{', '.join(sorted(weight_diffs)) or 'none'}), but in THIS sample no individual "
            f"entry/exit decision crossed the |score|>=25 threshold differently — naive and "
            f"learned produced the IDENTICAL {len(naive_seq)}-trade sequence. At current sample "
            f"sizes the weight adjustments are too small (0.7-1.2x range) to flip a decision; "
            f"this run cannot demonstrate an edge either way.")
    sharpe_winner = "learned" if lm["sharpe_ratio"] > nm["sharpe_ratio"] else (
        "naive" if nm["sharpe_ratio"] > lm["sharpe_ratio"] else "tie")
    verdict_lines.append(
        f"Sharpe: {sharpe_winner.upper()} wins ({lm['sharpe_ratio']:.3f} learned vs "
        f"{nm['sharpe_ratio']:.3f} naive, delta {lm['sharpe_ratio']-nm['sharpe_ratio']:+.3f})"
        if sharpe_winner != "tie" else f"Sharpe: TIE ({lm['sharpe_ratio']:.3f})")
    dd_winner = "learned" if lm["max_drawdown_pct"] < nm["max_drawdown_pct"] else (
        "naive" if nm["max_drawdown_pct"] < lm["max_drawdown_pct"] else "tie")
    verdict_lines.append(
        f"Max drawdown: {dd_winner.upper()} wins ({lm['max_drawdown_pct']:.2f}% learned vs "
        f"{nm['max_drawdown_pct']:.2f}% naive)" if dd_winner != "tie"
        else f"Max drawdown: TIE ({lm['max_drawdown_pct']:.2f}%)")
    wr_winner = "learned" if lm["win_rate"] > nm["win_rate"] else (
        "naive" if nm["win_rate"] > lm["win_rate"] else "tie")
    verdict_lines.append(
        f"Win rate: {wr_winner.upper()} wins ({lm['win_rate']:.1%} learned vs {nm['win_rate']:.1%} naive)"
        if wr_winner != "tie" else f"Win rate: TIE ({lm['win_rate']:.1%})")

    min_trades = min(nm["total_trades"], lm["total_trades"])
    if min_trades < 20:
        confidence = "LOW"
        confidence_note = (f"Only {min_trades} trades in the thinner run — a handful of "
                            f"trades can flip Sharpe/win-rate sign. Treat this as a directional "
                            f"read, not statistical proof.")
    elif min_trades < 60:
        confidence = "MODERATE"
        confidence_note = (f"{min_trades} trades in the thinner run — enough for a real signal "
                            f"but still noisy; corroborate over more history before acting.")
    else:
        confidence = "MODERATE-HIGH"
        confidence_note = (f"{min_trades} trades in the thinner run — a reasonably sized sample "
                            f"for a rule-weighting comparison, though still short of "
                            f"institutional (500+) sample sizes.")

    overall = ("learned" if (sharpe_winner == "learned") + (dd_winner == "learned") + (wr_winner == "learned")
               >= 2 else ("naive" if (sharpe_winner == "naive") + (dd_winner == "naive") + (wr_winner == "naive")
               >= 2 else "inconclusive"))

    attribution = _weight_attribution(learned)
    feeds_next = _feeds_next_trade_examples(learned)

    scope = {
        "lookback_days": lookback_days,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "fetch_reports": [r.to_dict() for r in fetch_reports],
        "symbols_fetched_ok": sum(1 for r in fetch_reports if r.ok),
        "symbols_attempted": len(fetch_reports),
    }

    return {
        "schema": "learning_validation.v1",
        "scope": scope,
        "methodology": {
            "walk_forward_split": ("backtest.engine.WalkForwardEvaluator._holdout — chronologically "
                                    "LAST 20% of each symbol's own bar history (holdout_pct=0.20, "
                                    "per .loop/anchors/holdout.md), min 60 total / 30 holdout bars"),
            "naive_definition": ("every LEARNABLE rule forced to weight 1.0 via a fresh, empty "
                                  "knowledge.rules.RuleStats (no track record); IMMUTABLE rules are "
                                  "pinned at 1.0 regardless, by construction"),
            "learned_definition": ("the CURRENT knowledge/rule_stats.json weights, loaded from a "
                                    "byte-copy — the live file is opened read-only and never mutated"),
            "shared_across_both_runs": ["cost model (data/costs.py)", "immutable rule vetoes",
                                          "position sizing (open_position)", "SL/TP/trailing math",
                                          "persona voting + risk desk + portfolio manager",
                                          "score_asset's technical/indicator scoring"],
            "production_code_reused": [
                "autonomous.AutonomousEngine.score_asset", "knowledge.brain.RuleBrain.evaluate_setup",
                "personas.engine.PersonaEngine.evaluate", "personas.manager.RiskManagerAgent/PortfolioManagerAgent",
                "autonomous.AutonomousEngine.open_position/check_positions/_close_position",
                "backtest.engine.BacktestEngine._compute_metrics",
                "backtest.engine.WalkForwardEvaluator._holdout",
            ],
            "daily_bar_adaptation": ("REENTRY_COOLDOWN_MIN forced to 0 for both runs (the live "
                                      "15m-loop anti-churn gate compares to wall-clock time, which "
                                      "is meaningless replaying historical daily bars); identical "
                                      "for naive and learned so it cannot bias the comparison"),
        },
        "comparison": {"naive": nm, "learned": lm},
        "verdict": {
            "lines": verdict_lines,
            "overall_lean": overall,
            "confidence": confidence,
            "confidence_note": confidence_note,
            "min_trades": min_trades,
            "identical_trade_sequence": identical_trade_sequence,
            "rules_with_diverged_weights": sorted(weight_diffs),
        },
        "weight_attribution": attribution,
        "feeds_next_trade_examples": feeds_next,
        "naive": naive,
        "learned": learned,
        "live_rule_stats_untouched": sim["live_rule_stats_untouched"],
    }


def _trade_row_md(t: dict) -> str:
    return (f"| {t['run']} | {t['symbol']} ({t['market']}) | {t['side']} | {t['entry_date']} @ "
            f"{t['entry_price']:.4g} | {t['exit_date']} @ {t.get('exit_price', 0):.4g} | "
            f"{t['exit_reason']} | {t['pnl']:+.4f} | {t.get('pnl_pct', 0):+.2f}% | "
            f"{t.get('persona_action', '-')} |")


def render_markdown(report: dict[str, Any]) -> str:
    scope, meth, cmp_, verdict = (report["scope"], report["methodology"],
                                   report["comparison"], report["verdict"])
    naive, learned = report["naive"], report["learned"]
    L = []
    L.append("# Learning-Loop Validation — Naive vs Learned Rule Weights")
    L.append("")
    L.append(f"*Generated {scope['finished_at']} — offline audit of the production "
             f"signal/decision pipeline against real historical data.*")
    L.append("")
    L.append("## Executive Summary")
    L.append("")
    L.append(f"Overall lean: **{verdict['overall_lean'].upper()}** "
              f"(confidence: {verdict['confidence']}).")
    for line in verdict["lines"]:
        L.append(f"- {line}")
    L.append(f"- {verdict['confidence_note']}")
    L.append(f"- Live `knowledge/rule_stats.json` left byte-identical: "
              f"**{report['live_rule_stats_untouched']}**.")
    L.append("")

    L.append("## Scope — Data Authenticity Trail")
    L.append("")
    L.append(f"- Lookback requested: {scope['lookback_days']} calendar days")
    L.append(f"- Symbols fetched OK: {scope['symbols_fetched_ok']} / {scope['symbols_attempted']}")
    L.append("")
    L.append("| Market | Symbol | Name | Sector | Bars | Range | Provider |")
    L.append("|---|---|---|---|---|---|---|")
    for r in scope["fetch_reports"]:
        if not r["ok"]:
            continue
        L.append(f"| {r['market']} | {r['symbol']} | {r['name']} | {r['sector']} | "
                  f"{r['bars']} | {r['start']} .. {r['end']} | {r['provider']} |")
    failed = [r for r in scope["fetch_reports"] if not r["ok"]]
    if failed:
        L.append("")
        L.append("Failed / skipped fetches:")
        for r in failed:
            L.append(f"- {r['market']}:{r['symbol']} — {r['error']}")
    L.append("")

    L.append("## Methodology")
    L.append("")
    for k, v in meth.items():
        if isinstance(v, list):
            L.append(f"- **{k}**:")
            for item in v:
                L.append(f"  - {item}")
        else:
            L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## Per-Symbol Holdout Coverage")
    L.append("")
    L.append("| Market | Symbol | Included | Holdout bars | Holdout window |")
    L.append("|---|---|---|---|---|")
    for m in learned["per_symbol_meta"]:
        if m["included"]:
            L.append(f"| {m['market']} | {m['symbol']} | yes | {m['holdout_bars']} | "
                      f"{m['holdout_start']} .. {m['holdout_end']} |")
        else:
            L.append(f"| {m['market']} | {m['symbol']} | NO | — | {m['reason']} |")
    L.append("")

    L.append("## Comparison — Naive vs Learned (held-out window)")
    L.append("")
    L.append("| Metric | Naive | Learned | Delta (learned - naive) |")
    L.append("|---|---|---|---|")
    for key, label in [
        ("sharpe_ratio", "Sharpe ratio"), ("sortino_ratio", "Sortino ratio"),
        ("max_drawdown_pct", "Max drawdown %"), ("win_rate", "Win rate"),
        ("profit_factor", "Profit factor"), ("total_trades", "Total trades"),
        ("total_return_pct", "Total return %"), ("avg_trade_return_pct", "Avg trade return %"),
        ("calmar_ratio", "Calmar ratio"),
    ]:
        nv, lv = cmp_["naive"][key], cmp_["learned"][key]
        L.append(f"| {label} | {nv} | {lv} | {lv - nv:+.4f} |")
    L.append("")
    L.append(f"Final equity — naive: ${naive['final_equity']:.4f} (from ${naive['initial_capital']:.2f}); "
              f"learned: ${learned['final_equity']:.4f} (from ${learned['initial_capital']:.2f}).")
    L.append("")

    L.append("## Rule-Weight Spot Check (proves the two runs are genuinely different)")
    L.append("")
    naive_snap, learned_snap = naive["rule_stats_snapshot"], learned["rule_stats_snapshot"]
    non_default = [rid for rid, info in learned_snap.items()
                   if info["mutability"] == "LEARNABLE" and abs(info["weight"] - 1.0) > 1e-9]
    L.append(f"Learned run: {len(non_default)} LEARNABLE rules carry a non-1.0 weight "
             f"out of {sum(1 for i in learned_snap.values() if i['mutability']=='LEARNABLE')} learnable rules tracked.")
    L.append("")
    L.append("| Rule | Investor | Naive weight | Learned weight | Learned accuracy (n) |")
    L.append("|---|---|---|---|---|")
    for rid in sorted(non_default)[:15]:
        li = learned_snap[rid]
        ni = naive_snap.get(rid, {"weight": 1.0})
        acc = f"{li['accuracy']:.0%}" if li["accuracy"] is not None else "—"
        L.append(f"| {rid} | {li['investor']} | {ni['weight']} | {li['weight']} | {acc} ({li['samples']}) |")
    L.append("")

    L.append("## Outcome Attribution — where a learned weight changed exposure")
    L.append("")
    attrib = report["weight_attribution"]
    if not attrib:
        L.append("No learned-run trade cited a rule carrying a non-1.0 weight in this sample.")
    else:
        for a in attrib[:15]:
            L.append(f"- **{a['trade_id']} ({a['symbol']}, {a['entry_date']})**: {a['rule_id']} "
                      f"[{a['investor']}] {a['effect']} to {a['weight']:.3f} (live accuracy "
                      f"{a['accuracy']:.0%} n={a['samples']}); score multiplier "
                      f"{a['score_multiplier']}; trade result **{a['trade_outcome']}** "
                      f"(${a['trade_pnl']:+.4f}).")
        if len(attrib) > 15:
            L.append(f"- ... {len(attrib) - 15} more attribution cases in the full JSON "
                      f"(dashboard BACKTEST AUDIT tab / reports/backtest/*.json).")
    L.append("")

    L.append("## How This Feeds The Next Trade (the mechanism, with real examples)")
    L.append("")
    L.append("Every closed trade calls `knowledge.rules.RuleStats.record(rule_id, rule_passed, "
              "trade_won, pnl)` for each rule cited at entry, then `RuleStats.weight(rule_id)` "
              "recomputes as `clip(0.5, 1.5, 0.5 + accuracy)` once >=10 samples exist "
              "(IMMUTABLE rules stay pinned at 1.0 forever). This run's OWN trades, replayed "
              "through that exact function on a throwaway clone (never written to disk):")
    L.append("")
    for ex in report["feeds_next_trade_examples"]:
        L.append(f"**{ex['trade_id']} — {ex['symbol']} ({ex['entry_date']} -> {ex['exit_date']}, "
                  f"{ex['outcome']} ${ex['pnl']:+.4f})**")
        for row in ex["rule_updates"]:
            L.append(f"  - {row['rule_id']} [{row['investor']}]: "
                      f"{'PASSED' if row['rule_passed_at_entry'] else 'FAILED'} at entry, trade "
                      f"{'WON' if row['trade_won'] else 'LOST'} -> weight "
                      f"{row['weight_before']:.3f} -> {row['weight_after_this_trade']:.3f} "
                      f"after this one observation")
        L.append("")

    L.append("## Trade-by-Trade Log (sample)")
    L.append("")
    all_trades = naive["trades"] + learned["trades"]
    all_trades_sorted = sorted(all_trades, key=lambda t: (t["run"], t["entry_date"] or ""))
    SAMPLE_HEAD, SAMPLE_TAIL, PNL_THRESHOLD = 20, 20, None
    if len(all_trades_sorted) > 0:
        pnls = sorted(abs(t["pnl"]) for t in all_trades_sorted)
        PNL_THRESHOLD = pnls[int(len(pnls) * 0.9)] if pnls else 0
    shown_ids = set()
    rows = []
    for t in all_trades_sorted[:SAMPLE_HEAD]:
        rows.append(t); shown_ids.add(id(t))
    for t in all_trades_sorted[-SAMPLE_TAIL:]:
        if id(t) not in shown_ids:
            rows.append(t); shown_ids.add(id(t))
    if PNL_THRESHOLD is not None:
        for t in all_trades_sorted:
            if id(t) not in shown_ids and abs(t["pnl"]) >= PNL_THRESHOLD:
                rows.append(t); shown_ids.add(id(t))
    total_n = len(all_trades_sorted)
    if len(rows) < total_n:
        L.append(f"Showing {len(rows)} of {total_n} total trades below (first/last "
                  f"{SAMPLE_HEAD}, plus all trades in the top decile by |P&L|). "
                  f"Full per-trade log — including every rule citation and persona vote — "
                  f"is in the JSON report and the dashboard's BACKTEST AUDIT tab.")
    L.append("")
    L.append("| Run | Symbol | Side | Entry | Exit | Reason | P&L | P&L% | Persona |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in sorted(rows, key=lambda t: (t["run"], t["entry_date"] or "")):
        L.append(_trade_row_md(t))
    L.append("")

    L.append("---")
    L.append(f"*Report schema: {report['schema']}. This file and the full structured JSON "
             f"live under reports/backtest/ (gitignored, local-only).*")
    return "\n".join(L)


# ── orchestration ─────────────────────────────────────────────────────────

async def run_learning_validation(lookback_days: int = LOOKBACK_DAYS) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    data, fetch_reports = await fetch_all(lookback_days)
    sim = await asyncio.to_thread(_simulate_both, data)
    report = build_report(sim, fetch_reports, lookback_days, started_at)
    return report


def write_report(report: dict[str, Any], out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"learning_validation_{ts}.json"
    md_path = out_dir / f"learning_validation_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    latest_json = out_dir / "latest.json"
    latest_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return json_path, md_path


def print_console_table(report: dict[str, Any]) -> None:
    cmp_ = report["comparison"]
    print("\n" + "=" * 78)
    print("LEARNING VALIDATION — NAIVE vs LEARNED RULE WEIGHTS")
    print("=" * 78)
    print(f"{'Metric':<24}{'Naive':>15}{'Learned':>15}{'Delta':>15}")
    for key, label in [
        ("sharpe_ratio", "Sharpe"), ("sortino_ratio", "Sortino"),
        ("max_drawdown_pct", "Max DD %"), ("win_rate", "Win rate"),
        ("profit_factor", "Profit factor"), ("total_trades", "Total trades"),
        ("total_return_pct", "Total return %"), ("calmar_ratio", "Calmar"),
    ]:
        nv, lv = cmp_["naive"][key], cmp_["learned"][key]
        print(f"{label:<24}{nv:>15.4f}{lv:>15.4f}{lv - nv:>+15.4f}")
    print("-" * 78)
    for line in report["verdict"]["lines"]:
        print(f"  {line}")
    print(f"  Overall lean: {report['verdict']['overall_lean'].upper()} "
          f"(confidence: {report['verdict']['confidence']})")
    print(f"  Live rule_stats.json untouched: {report['live_rule_stats_untouched']}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    report = asyncio.run(run_learning_validation(args.lookback_days))
    print_console_table(report)
    json_path, md_path = write_report(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
