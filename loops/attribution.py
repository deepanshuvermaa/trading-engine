"""Post-trade attribution — spec §9.1: "which factor was wrong".

On trade close, decomposes realized P&L into deterministic components:

  thesis_pnl   — gross directional move x size: was the *idea* right?
  timing_pnl   — realized P&L minus thesis P&L: execution drag
                 (slippage + commissions + fill quality)
  sizing_error — realized_R x (actual_risk - reference_risk):
                 over/under-bet vs the V8/Kelly reference
  exit_efficiency_atr — (exit vs target) in ATRs: how much was given back
  regime_penalty — trade return minus this module's average return in the
                 same regime bucket (trending/ranging by entry ADX):
                 did the signal fire in the wrong regime?

Records go to reports/audit/attribution.jsonl. Per-rule win/loss stats go to
knowledge/rule_stats.json so the outer loop can down-weight rules that keep
failing — LEARNABLE rules only; IMMUTABLE risk vetoes may never be relaxed
(enforced inside RuleStats.weight).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from db.store import get_store
from knowledge.rules import RuleStats
from utils.logger import get_logger

log = get_logger("loop.attribution")

ATTRIBUTION_PATH = Path("./reports/audit/attribution.jsonl")

# V8 reference risk (Minervini): the sizing benchmark for the sizing-error term
REFERENCE_RISK_PCT = 1.25


class AttributionRecord(BaseModel):
    """One closed trade, fully decomposed."""

    trade_id: str
    symbol: str
    module: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    ret_pct: float = 0.0  # pnl as % of equity at close
    exit_reason: str
    opened_at: str
    closed_at: str
    regime: str  # "trending" | "ranging" | "unknown" (entry ADX bucket)
    # decomposition
    thesis_pnl: float
    timing_pnl: float
    sizing_error: float
    exit_efficiency_atr: float | None
    regime_penalty: float | None
    realized_r: float | None
    # causal record
    rule_citations: list[str] = Field(default_factory=list)
    rules_passed: list[str] = Field(default_factory=list)
    rules_failed: list[str] = Field(default_factory=list)
    premortem: list[dict] = Field(default_factory=list)  # M6: conditions that would make it wrong


class TradeAttribution:
    """Entry snapshots + close-time attribution + per-rule stats maintenance."""

    def __init__(
        self,
        path: str | Path = ATTRIBUTION_PATH,
        rule_stats: RuleStats | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rule_stats = rule_stats or RuleStats()
        self._open: dict[str, dict] = {}  # trade_id -> entry snapshot
        # regime-conditional running perf: (module, regime) -> {sum_ret, n}
        self._regime_perf: dict[str, dict[str, float]] = {}
        self._load_regime_perf()

    # ── entry ────────────────────────────────────────────────────────────

    def record_entry(self, trade_id: str, setup: dict, context: dict, verdict: Any = None) -> None:
        """Store the causal record at entry (spec §9.1 TradeRecord).

        `verdict` is the RuleVerdict from RuleBrain.evaluate_setup (optional).
        The pre-mortem kill-list (M6) is derived from the rules that FAILED at
        entry but did not veto: each is a measurable condition that argues the
        trade is wrong.
        """
        citations, passed, failed, premortem = [], [], [], []
        if verdict is not None:
            citations = list(verdict.citations)
            passed = [c.rule_id for c in verdict.passed]
            failed = [c.rule_id for c in verdict.failed]
            premortem = [
                {"condition": c.threshold, "rule_id": c.rule_id, "observed_at_entry": c.actual}
                for c in verdict.failed
            ]
        self._open[trade_id] = {
            "trade_id": trade_id,
            "setup": setup,
            "context": context,
            "rule_citations": citations,
            "rules_passed": passed,
            "rules_failed": failed,
            "premortem": premortem,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── close ────────────────────────────────────────────────────────────

    def on_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        size: float,
        entry_price: float,
    ) -> AttributionRecord | None:
        """Decompose realized P&L, persist, update per-rule win/loss stats."""
        snap = self._open.pop(trade_id, None)
        if snap is None:
            log.warning(f"attribution: no entry snapshot for {trade_id}; skipping decomposition")
            return None

        setup = snap["setup"]
        context = snap.get("context", {})
        side = setup.get("direction", "BUY")
        module = setup.get("module", "unknown")
        symbol = setup.get("symbol", trade_id)
        dir_sign = 1.0 if side == "BUY" else -1.0

        # 1. Thesis P&L — gross directional move (was the idea right?)
        thesis_pnl = (exit_price - entry_price) * size * dir_sign

        # 2. Timing P&L — execution drag (realized minus gross = slippage/fees/fills)
        timing_pnl = pnl - thesis_pnl

        # 3. Sizing error — realized_R x (actual_risk - reference_risk)
        sl = setup.get("stop_loss")
        risk_per_unit = abs(entry_price - float(sl)) if sl is not None else None
        realized_r = None
        sizing_error = 0.0
        equity = context.get("equity") or 0.0
        if risk_per_unit and risk_per_unit > 0 and equity > 0:
            risk_amount = size * risk_per_unit
            realized_r = pnl / risk_amount if risk_amount > 0 else None
            actual_risk_pct = risk_amount / equity * 100.0
            if realized_r is not None:
                sizing_error = realized_r * (actual_risk_pct - REFERENCE_RISK_PCT)

        # 4. Exit efficiency — distance from target in ATRs (give-back proxy)
        atr = setup.get("atr")
        tp = setup.get("take_profit")
        exit_efficiency_atr = None
        if atr and tp is not None and float(atr) > 0:
            exit_efficiency_atr = round((exit_price - float(tp)) * dir_sign / float(atr), 3)

        # 5. Regime penalty — trade return vs module's average in same regime bucket
        adx = setup.get("adx")
        regime = "unknown"
        if adx is not None:
            regime = "trending" if float(adx) > 25 else "ranging"
        ret_pct = pnl / equity * 100.0 if equity > 0 else 0.0
        key = f"{module}|{regime}"
        perf = self._regime_perf.get(key, {"sum_ret": 0.0, "n": 0})
        avg = perf["sum_ret"] / perf["n"] if perf["n"] > 0 else None
        regime_penalty = round(ret_pct - avg, 4) if avg is not None else None
        perf["sum_ret"] += ret_pct
        perf["n"] += 1
        self._regime_perf[key] = perf

        record = AttributionRecord(
            trade_id=trade_id,
            symbol=symbol,
            module=module,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=round(pnl, 6),
            ret_pct=round(ret_pct, 6),
            exit_reason=exit_reason,
            opened_at=snap.get("opened_at", ""),
            closed_at=datetime.now(timezone.utc).isoformat(),
            regime=regime,
            thesis_pnl=round(thesis_pnl, 6),
            timing_pnl=round(timing_pnl, 6),
            sizing_error=round(sizing_error, 6),
            exit_efficiency_atr=exit_efficiency_atr,
            regime_penalty=regime_penalty,
            realized_r=round(realized_r, 4) if realized_r is not None else None,
            rule_citations=snap["rule_citations"],
            rules_passed=snap["rules_passed"],
            rules_failed=snap["rules_failed"],
            premortem=snap["premortem"],
        )
        self._persist(record)
        self._update_rule_stats(record)
        return record

    # ── internals ────────────────────────────────────────────────────────

    def _update_rule_stats(self, record: AttributionRecord) -> None:
        """Credit/debit every rule that voted on this trade at entry.

        LEARNABLE rules earn/lose weight; IMMUTABLE rules are tracked for the
        record but RuleStats pins their weight at 1.0 — vetoes never relax.
        """
        won = record.pnl > 0
        for rule_id in record.rules_passed:
            self.rule_stats.record(rule_id, rule_passed=True, trade_won=won, pnl=record.pnl)
        for rule_id in record.rules_failed:
            self.rule_stats.record(rule_id, rule_passed=False, trade_won=won, pnl=record.pnl)
        self.rule_stats.save()

    def _persist(self, record: AttributionRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        # Write-through to Postgres when configured (JSONL is the fallback).
        store = get_store()
        if store.enabled:
            store.fire(store.save_attribution(record.model_dump()))

    async def load_regime_perf_from_db(self) -> bool:
        """Rebuild regime-conditional perf from the attribution table.

        Called on engine startup after Store.init(). The DB aggregate is
        authoritative when available (survives ephemeral filesystems);
        otherwise the JSONL rebuild from __init__ stands.
        """
        store = get_store()
        if not store.enabled:
            return False
        try:
            perf = await store.load_regime_perf()
        except Exception as e:
            log.warning(f"attribution: DB regime-perf rebuild failed: {e}")
            return False
        if not perf:
            return False
        self._regime_perf = perf
        log.info(f"attribution: regime perf rebuilt from Postgres "
                 f"({len(perf)} module|regime buckets)")
        return True

    def _load_regime_perf(self) -> None:
        """Rebuild regime-conditional perf from the attribution log on restart."""
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = f"{rec.get('module', 'unknown')}|{rec.get('regime', 'unknown')}"
                    perf = self._regime_perf.setdefault(key, {"sum_ret": 0.0, "n": 0})
                    perf["sum_ret"] += float(rec.get("ret_pct", rec.get("pnl", 0.0)))
                    perf["n"] += 1
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"attribution: could not rebuild regime perf: {e}")
