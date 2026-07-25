"""RuleBrain — the "accurate access embedded" layer.

Given a candidate trade + market/portfolio context, evaluates every applicable
investor rule deterministically (no LLM, no randomness) and returns pass/fail
per rule with a citation: rule id + investor + exact threshold vs actual value.

Rules whose required inputs are absent (e.g. 10y fundamentals, 13F data) are
reported as not-applicable and excluded from the verdict — the brain never
guesses. IMMUTABLE rule failures are hard vetoes; LEARNABLE failures shade the
score via learned rule weights (knowledge/rule_stats.json).
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from knowledge.rules import Mutability, Rule, RuleBook, RuleStats, load_rules
from utils.logger import get_logger

log = get_logger("knowledge.brain")

# An evaluator inspects (trade, context) and returns (passed, actual) or None
# when the rule does not apply / data is missing.
Evaluator = Callable[[Rule, dict, dict], "tuple[bool, str] | None"]


class RuleCheck(BaseModel):
    """Result of evaluating one rule against one candidate trade."""

    rule_id: str
    investor: str
    passed: bool
    mutability: Mutability
    threshold: str
    actual: str
    weight: float = 1.0

    @property
    def citation(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict} [{self.rule_id}|{self.investor}] threshold: {self.threshold} | actual: {self.actual}"


class RuleVerdict(BaseModel):
    """Aggregate verdict for a candidate trade."""

    checks: list[RuleCheck]
    vetoed: bool
    veto_citations: list[str]
    score_multiplier: float  # applied to the setup score (LEARNABLE rules only)

    @property
    def citations(self) -> list[str]:
        return [c.citation for c in self.checks]

    @property
    def passed(self) -> list[RuleCheck]:
        return [c for c in self.checks if c.passed]

    @property
    def failed(self) -> list[RuleCheck]:
        return [c for c in self.checks if not c.passed]


# ── helpers ──────────────────────────────────────────────────────────────

def _num(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v


def _stop_pct(trade: dict) -> float | None:
    price, sl = _num(trade, "current_price"), _num(trade, "stop_loss")
    if price is None or sl is None or price <= 0:
        return None
    return abs(price - sl) / price * 100.0


# ── evaluators (deterministic; None = not applicable / no data) ─────────

def _eval_L1(rule: Rule, t: dict, c: dict):
    price, s200 = _num(t, "current_price"), _num(t, "sma_200")
    if price is None or not s200:
        return None
    if t.get("direction") == "BUY":
        return price > s200, f"BUY with price {price:.2f} vs 200d MA {s200:.2f}"
    return price < s200, f"SELL with price {price:.2f} vs 200d MA {s200:.2f}"


def _eval_L9(rule: Rule, t: dict, c: dict):
    unreal = c.get("existing_position_unrealized")
    if unreal is None:
        return None  # no existing position on this symbol — rule not in play
    return unreal >= 0, f"existing position unrealized P&L = {unreal:+.4f}"


def _eval_L10(rule: Rule, t: dict, c: dict):
    sp = _stop_pct(t)
    if sp is None:
        return None
    return sp <= rule.params.get("max_loss_pct", 10), f"stop distance {sp:.2f}% of price"


def _eval_V7(rule: Rule, t: dict, c: dict):
    sp = _stop_pct(t)
    if sp is None:
        return None
    lim = rule.params.get("stop_max_pct", 10)
    return sp <= lim, f"initial stop {sp:.2f}% below entry (ceiling {lim}%)"


def _eval_CH10(rule: Rule, t: dict, c: dict):
    if t.get("module") != "breakout":
        return None  # cardinal rule of the breakout sleeve only
    sp = _stop_pct(t)
    if sp is None:
        return None
    lim = rule.params.get("stop_max_pct", 8)
    return sp <= lim, f"breakout-sleeve stop {sp:.2f}% (hard ceiling {lim}%)"


def _eval_V8(rule: Rule, t: dict, c: dict):
    risk = _num(c, "planned_risk_pct")
    if risk is None:
        return None
    lim = rule.params.get("risk_pct_max", 1.25)
    return risk <= lim, f"planned risk {risk:.2f}% of equity (cap {lim}%)"


def _eval_V9(rule: Rule, t: dict, c: dict):
    notional = _num(c, "planned_notional_pct")
    if notional is None:
        return None
    lim = rule.params.get("position_max_pct", 25)
    return notional <= lim, f"planned position {notional:.1f}% of book (cap {lim}%)"


def _eval_V10(rule: Rule, t: dict, c: dict):
    rr = _num(t, "risk_reward")
    if rr is None:
        return None
    lim = rule.params.get("rr_min", 2.0)
    return rr >= lim, f"reward:risk {rr:.2f} (floor {lim}:1)"


def _eval_D4(rule: Rule, t: dict, c: dict):
    if t.get("module") != "trend_follower":
        return None  # macro-directional sleeve only
    rr = _num(t, "risk_reward")
    if rr is None:
        return None
    lim = rule.params.get("rr_min", 3.0)
    return rr >= lim, f"macro-sleeve reward:risk {rr:.2f} (floor {lim}:1)"


def _eval_D7(rule: Rule, t: dict, c: dict):
    adx, price, s200 = _num(t, "adx"), _num(t, "current_price"), _num(t, "sma_200")
    if adx is None or price is None or not s200:
        return None
    dist = (price - s200) / s200 * 100.0
    strong_up = adx > rule.params.get("adx_min", 30) and dist > rule.params.get("ma_dist_pct", 20)
    strong_dn = adx > rule.params.get("adx_min", 30) and dist < -rule.params.get("ma_dist_pct", 20)
    if strong_up and t.get("direction") == "SELL":
        return False, f"counter-trend SELL: ADX {adx:.0f}, price {dist:+.1f}% vs 200d MA"
    if strong_dn and t.get("direction") == "BUY":
        return False, f"counter-trend BUY: ADX {adx:.0f}, price {dist:+.1f}% vs 200d MA"
    return True, f"ADX {adx:.0f}, price {dist:+.1f}% vs 200d MA — not fighting a strong trend"


def _eval_D8(rule: Rule, t: dict, c: dict):
    dd = _num(c, "drawdown_pct")
    if dd is None:
        return None
    flat = rule.params.get("flat_dd_pct", 10)
    cut = rule.params.get("cut_dd_pct", 5)
    if dd >= flat:
        return False, f"drawdown {dd:.2f}% >= {flat}% — mandatory flat + cooloff"
    note = " (cut-gross-50% zone)" if dd >= cut else ""
    return True, f"drawdown {dd:.2f}%{note}"


def _eval_D9(rule: Rule, t: dict, c: dict):
    gross = _num(c, "gross_exposure_pct")
    new = _num(c, "planned_notional_pct") or 0.0
    ytd = _num(c, "total_return_pct")
    if gross is None or ytd is None:
        return None
    if ytd > rule.params.get("ytd_up", 10):
        cap = rule.params.get("up_cap", 150)
    elif ytd < 0:
        cap = rule.params.get("down_cap", 70)
    else:
        cap = rule.params.get("base", 100)
    total = gross + new
    return total <= cap, f"gross after entry {total:.1f}% vs cap {cap}% (YTD {ytd:+.2f}%)"


def _eval_M7(rule: Rule, t: dict, c: dict):
    reasons = t.get("reasons")
    if reasons is None:
        return None
    n = len(reasons)
    need = int(rule.params.get("min_factors", 3))
    return n >= need, f"{n} independent passing factors (need >= {need} for full size)"


def _eval_V12(rule: Rule, t: dict, c: dict):
    wr = _num(c, "last10_win_rate")
    n = c.get("last10_n", 0)
    if wr is None or n < rule.params.get("lookback", 10):
        return None
    stop_below = rule.params.get("stop_below", 0.30)
    halve_below = rule.params.get("halve_below", 0.40)
    if wr < stop_below:
        return False, f"last-{n} win rate {wr:.0%} < {stop_below:.0%} — stop trading, paper-trade"
    note = " (half-size zone)" if wr < halve_below else ""
    return True, f"last-{n} win rate {wr:.0%}{note}"


def _eval_K4(rule: Rule, t: dict, c: dict):
    stats = c.get("module_stats") or {}
    trades = stats.get("trades", 0)
    if trades < 50:
        return None  # K1 requires min 50 trades of rolling stats
    wins = stats.get("wins", 0)
    p = wins / trades
    q = 1 - p
    b = _num(t, "risk_reward") or 0.0
    if b <= 0:
        return None
    # K5 shrinkage: shrink p 50% toward 0.5 before Kelly (estimation-error guard)
    p_shrunk = 0.5 + (p - 0.5) * 0.5
    f_star = (b * p_shrunk - (1 - p_shrunk)) / b
    frac = (_num(c, "planned_risk_pct") or 0.0) / 100.0
    if f_star <= 0:
        return False, f"shrunk Kelly f*={f_star:.3f} <= 0 for module '{t.get('module')}' — no edge"
    cap = rule.params.get("hard_cap", 0.5) * f_star
    return frac <= cap, f"risk fraction {frac:.4f} vs 0.5-Kelly cap {cap:.4f} (f*={f_star:.3f})"


def _eval_T1(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None  # trend template qualifies longs
    price, s150, s200 = _num(t, "current_price"), _num(t, "sma_150"), _num(t, "sma_200")
    if price is None or not s150 or not s200:
        return None
    return price > s150 and price > s200, f"price {price:.2f} vs 150d {s150:.2f} / 200d {s200:.2f}"


def _eval_T2(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None
    s150, s200 = _num(t, "sma_150"), _num(t, "sma_200")
    if not s150 or not s200:
        return None
    return s150 > s200, f"150d MA {s150:.2f} vs 200d MA {s200:.2f}"


def _eval_T3(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None
    s200, s200_prev = _num(t, "sma_200"), _num(t, "sma_200_1m_ago")
    if not s200 or not s200_prev:
        return None
    return s200 > s200_prev, f"200d MA {s200:.2f} vs 1 month ago {s200_prev:.2f}"


def _eval_T4(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None
    s50, s150, s200 = _num(t, "sma_50"), _num(t, "sma_150"), _num(t, "sma_200")
    if not s50 or not s150 or not s200:
        return None
    return s50 > s150 and s50 > s200, f"50d {s50:.2f} vs 150d {s150:.2f} / 200d {s200:.2f}"


def _eval_T5(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None
    price, s50 = _num(t, "current_price"), _num(t, "sma_50")
    if price is None or not s50:
        return None
    return price > s50, f"price {price:.2f} vs 50d MA {s50:.2f}"


def _eval_T6(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None
    price, lo = _num(t, "current_price"), _num(t, "low_52w")
    if price is None or not lo:
        return None
    mult = rule.params.get("low_mult", 1.30)
    return price >= mult * lo, f"price {price:.2f} = {price / lo:.2f}x 52w low {lo:.2f} (need {mult}x)"


def _eval_T7(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY":
        return None
    price, hi = _num(t, "current_price"), _num(t, "high_52w")
    if price is None or not hi:
        return None
    mult = rule.params.get("high_mult", 0.75)
    return price >= mult * hi, f"price {price:.2f} = {price / hi:.0%} of 52w high {hi:.2f} (need {mult:.0%})"


def _eval_CSN(rule: Rule, t: dict, c: dict):
    if t.get("direction") != "BUY" or t.get("module") != "breakout":
        return None
    price, hi = _num(t, "current_price"), _num(t, "high_52w")
    if price is None or not hi:
        return None
    dist = (hi - price) / hi * 100.0
    lim = rule.params.get("high_dist_pct", 15)
    return dist <= lim, f"price {dist:.1f}% below 52w high (need within {lim}%)"


def _eval_V5(rule: Rule, t: dict, c: dict):
    if t.get("module") != "breakout":
        return None
    vr = _num(t, "volume_ratio")
    if vr is None:
        return None
    lim = rule.params.get("vol_mult", 1.5)
    return vr >= lim, f"breakout-day volume {vr:.2f}x avg (need >= {lim}x)"


def _eval_S10(rule: Rule, t: dict, c: dict):
    price, atr = _num(t, "current_price"), _num(t, "atr")
    vr = _num(t, "volume_ratio")
    if price is None or atr is None:
        return None
    ok = price > 0 and atr > 0 and (vr is None or vr >= 0)
    return ok, f"bar sanity: price {price:.2f}, ATR {atr:.4f}, vol_ratio {vr}"


EVALUATORS: dict[str, Evaluator] = {
    "L1": _eval_L1,
    "L9": _eval_L9,
    "L10": _eval_L10,
    "V7": _eval_V7,
    "CH10": _eval_CH10,
    "V8": _eval_V8,
    "V9": _eval_V9,
    "V10": _eval_V10,
    "D4": _eval_D4,
    "D7": _eval_D7,
    "D8": _eval_D8,
    "D9": _eval_D9,
    "M7": _eval_M7,
    "V12": _eval_V12,
    "K4": _eval_K4,
    "T1": _eval_T1,
    "T2": _eval_T2,
    "T3": _eval_T3,
    "T4": _eval_T4,
    "T5": _eval_T5,
    "T6": _eval_T6,
    "T7": _eval_T7,
    "CS-N": _eval_CSN,
    "V5": _eval_V5,
    "S10": _eval_S10,
}


class RuleBrain:
    """Evaluates every applicable investor rule against a candidate trade.

    Deterministic only — no LLM in the trading loop. Every decision carries
    citations (rule id + investor + threshold vs actual).
    """

    def __init__(self, rulebook: RuleBook | None = None, stats: RuleStats | None = None):
        self.rulebook = rulebook or load_rules()
        self.stats = stats or RuleStats(rulebook=self.rulebook)

    def evaluate_setup(self, trade: dict, context: dict | None = None) -> RuleVerdict:
        """Evaluate all applicable rules. Returns verdict with per-rule citations."""
        context = context or {}
        checks: list[RuleCheck] = []

        for rule in self.rulebook.all():
            fn = EVALUATORS.get(rule.id)
            if fn is None:
                continue  # no deterministic evaluator / data source wired yet
            result = fn(rule, trade, context)
            if result is None:
                continue  # rule not applicable to this trade / missing inputs
            passed, actual = result
            checks.append(
                RuleCheck(
                    rule_id=rule.id,
                    investor=rule.investor,
                    passed=passed,
                    mutability=rule.mutability,
                    threshold=rule.threshold,
                    actual=actual,
                    weight=self.stats.weight(rule.id),
                )
            )

        veto_checks = [c for c in checks if not c.passed and c.mutability == Mutability.IMMUTABLE]
        multiplier = self._score_multiplier(checks)
        return RuleVerdict(
            checks=checks,
            vetoed=bool(veto_checks),
            veto_citations=[c.citation for c in veto_checks],
            score_multiplier=multiplier,
        )

    def _score_multiplier(self, checks: list[RuleCheck]) -> float:
        """Weighted pass-ratio of LEARNABLE checks -> score multiplier in [0.7, 1.2].

        Rule weights come from knowledge/rule_stats.json: rules that keep being
        wrong contribute less (the outer loop's down-weighting in action).
        """
        learnable = [c for c in checks if c.mutability == Mutability.LEARNABLE]
        if not learnable:
            return 1.0
        total_w = sum(c.weight for c in learnable)
        if total_w <= 0:
            return 1.0
        passed_w = sum(c.weight for c in learnable if c.passed)
        ratio = passed_w / total_w
        return round(max(0.7, min(1.2, 0.7 + 0.5 * ratio)), 4)

    def summary(self) -> dict:
        """Registry overview — used by verification and the dashboard."""
        book = self.rulebook
        return {
            "rules_loaded": len(book.rules),
            "immutable": len(book.immutable()),
            "learnable": len(book.learnable()),
            "evaluable": len([r for r in book.all() if r.id in EVALUATORS]),
            "investors": {inv: len(book.by_investor(inv)) for inv in book.investors()},
            "immutable_ids": sorted(r.id for r in book.immutable()),
            "stats_tracked": len(self.stats.stats),
            "source": book.source,
        }
