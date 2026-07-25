"""PersonaEngine — ten codified investor personas vote on every setup.

Architecture stolen from virattt/ai-hedge-fund (investor-persona agents that
each emit signal + confidence + reasoning, synthesized by a portfolio
manager) — but strictly deterministic. NO LLM anywhere in the loop:

- Each persona sees ONLY its own rules from knowledge/rules.yaml, evaluated
  by the shared RuleBrain (pass/fail citations with exact thresholds).
- On top of the rules, each persona applies an investor-specific scoring
  lens: a fixed arithmetic function of the setup's technical/macro fields
  (e.g. Livermore weights trend/structure, Minervini weights the trend
  template, Dalio weights regime/macro mood, Thorp votes only on Kelly edge).
- The one-line "quote" comes from a small deterministic template bank keyed
  by the persona's dominant factor. Same inputs, same vote, same words.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from db.store import get_store
from knowledge.brain import RuleBrain, RuleVerdict
from utils.logger import get_logger

log = get_logger("personas.engine")

PERSONA_STATS_PATH = Path(__file__).parent / "persona_stats.json"

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

# Lens verdict thresholds: |directional score| >= this -> a real opinion.
SIGNAL_THRESHOLD = 0.20


class PersonaVote(BaseModel):
    """One partner's opinion on one setup — fully deterministic."""

    persona: str                 # display name (the columnist byline)
    investor: str                # investor key in knowledge/rules.yaml
    signal: str = NEUTRAL        # BULLISH / BEARISH / NEUTRAL
    confidence: int = 0          # 0-100
    reasoning: list[str] = Field(default_factory=list)  # cited rules + lens notes
    quote: str = ""              # one-liner from the template bank
    dominant_factor: str = ""    # which lens factor drove the vote


# ── field helpers ─────────────────────────────────────────────────────────

def _num(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _dirsign(t: dict) -> int:
    return 1 if t.get("direction") == "BUY" else -1


# ── scoring lenses ────────────────────────────────────────────────────────
# Each lens: (setup, context, persona_rule_checks) -> (factors, notes)
# factors: name -> signed contribution in [-1, 1] toward BULLISH(+)/BEARISH(-)

Lens = Callable[[dict, dict, list], "tuple[dict[str, float], list[str]]"]


def _lens_buffett(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    price, s50, s200 = _num(t, "current_price"), _num(t, "sma_50"), _num(t, "sma_200")
    hi, rsi = _num(t, "high_52w"), _num(t, "rsi")
    if price and s200:
        if price > s200 and (not s50 or s50 > s200):
            f["durable_trend"] = 0.45
            notes.append("LENS: price holds above a rising long-term base — durability proxy")
        elif price < s200:
            f["durable_trend"] = -0.45
            notes.append("LENS: below the 200-day — the moat proxy is breached")
    if price and hi and hi > 0:
        disc = (hi - price) / hi
        if disc >= 0.10 and s200 and price > s200:
            f["margin_of_safety"] = min(0.40, disc * 1.5)
            notes.append(f"LENS: {disc:.0%} below the 52-week high with trend intact — a fair price for quality")
        elif disc < 0.02:
            f["priced_for_perfection"] = -0.20
            notes.append("LENS: at the highs — no discount on offer")
    if rsi is not None and rsi > 70:
        f["overpaying"] = -0.30
        notes.append(f"LENS: RSI {rsi:.0f} — the crowd is paying up")
    notes.append("LENS: 10-year fundamentals (B1-B16) not on file — voting on durability proxies only")
    return f, notes


def _lens_graham(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    rsi = _num(t, "rsi")
    price, lo, hi = _num(t, "current_price"), _num(t, "low_52w"), _num(t, "high_52w")
    if rsi is not None:
        if rsi <= 30:
            f["fear_discount"] = 0.50
            notes.append(f"LENS: RSI {rsi:.0f} — Mr. Market is despondent; that is when he quotes bargains")
        elif rsi >= 70:
            f["speculative_froth"] = -0.50
            notes.append(f"LENS: RSI {rsi:.0f} — enthusiasm, the enemy of the intelligent investor")
        else:
            f["price_temperature"] = round((50 - rsi) / 80, 3)
    if price and lo and hi and hi > lo:
        pos = (price - lo) / (hi - lo)
        if pos <= 0.25:
            f["near_52w_low"] = 0.35
            notes.append(f"LENS: price in the bottom quartile of its 52-week range ({pos:.0%})")
        elif pos >= 0.90:
            f["no_margin_of_safety"] = -0.35
            notes.append(f"LENS: price in the top decile of its range ({pos:.0%}) — no margin of safety")
    notes.append("LENS: no balance sheet on file (G/GR rules idle) — judging price against its own history")
    return f, notes


def _lens_livermore(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    reasons = t.get("reasons") or []
    price, s200 = _num(t, "current_price"), _num(t, "sma_200")
    adx, vr = _num(t, "adx"), _num(t, "volume_ratio")
    trend = 0.0
    if price and s200:
        trend = 0.50 if price > s200 else -0.50
        f["line_of_least_resistance"] = trend
        notes.append(f"LENS: the line of least resistance points {'up' if trend > 0 else 'down'} (price vs 200d)")
    if "Bullish structure break" in reasons:
        f["pivotal_point"] = 0.35
        notes.append("LENS: bullish market-structure break — a pivotal point printed")
    elif "Bearish structure break" in reasons:
        f["pivotal_point"] = -0.35
        notes.append("LENS: bearish market-structure break — the tape turned")
    if "FRESH bullish EMA cross" in reasons:
        f["fresh_move"] = 0.30
    elif "FRESH bearish EMA cross" in reasons:
        f["fresh_move"] = -0.30
    if vr is not None and vr >= 1.5 and trend:
        f["tape_volume"] = 0.15 * (1 if trend > 0 else -1)
        notes.append(f"LENS: volume {vr:.1f}x average confirms the move")
    if adx is not None and adx < 15:
        notes.append(f"LENS: ADX {adx:.0f} — a dull market; time to go fishing, not trading")
    return f, notes


def _lens_druckenmiller(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    sent = _num(t, "market_sentiment")
    price, s200, adx = _num(t, "current_price"), _num(t, "sma_200"), _num(t, "adx")
    rr = _num(t, "risk_reward")
    ds = _dirsign(t)
    if sent is not None and sent != 0:
        f["liquidity_mood"] = round(_clamp(sent) * 0.50, 3)
        notes.append(f"LENS: market-bucket news mood {sent:+.2f} — liquidity/sentiment tailwind" if sent > 0
                     else f"LENS: market-bucket news mood {sent:+.2f} — the liquidity wind is in our face")
    if price and s200 and adx is not None:
        f["macro_trend"] = round(0.40 * (1 if price > s200 else -1) * min(adx, 40) / 40, 3)
        notes.append(f"LENS: ADX {adx:.0f} trend, price {'above' if price > s200 else 'below'} the 200d")
    if rr is not None:
        if rr >= 3.0:
            f["asymmetry"] = 0.25 * ds
            notes.append(f"LENS: {rr:.1f}:1 payoff — the kind of asymmetry worth swinging at")
        elif rr < 2.0:
            f["poor_asymmetry"] = -0.20 * ds
            notes.append(f"LENS: {rr:.1f}:1 payoff — not enough asymmetry to bet big")
    return f, notes


def _lens_simons(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    s10 = next((ch for ch in checks if ch.rule_id == "S10"), None)
    if s10 is not None and not s10.passed:
        notes.append("LENS: bar failed data-sanity checks (S10) — quarantined, no signal fires")
        return f, notes
    score = _num(t, "score") or 0.0
    vr = _num(t, "volume_ratio")
    f["signal_strength"] = round(_clamp(score / 100.0) * 0.60, 3)
    notes.append(f"LENS: composite signal {score:+.0f}/100 — the model's residual edge")
    if vr is not None and vr >= 1.5 and score:
        f["volume_anomaly"] = 0.15 * (1 if score > 0 else -1)
        notes.append(f"LENS: volume anomaly {vr:.1f}x — statistically interesting participation")
    notes.append("LENS: one signal is not twenty (S1) — conviction capped accordingly")
    return f, notes


def _lens_dalio(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    sent = _num(t, "market_sentiment")
    price, s200 = _num(t, "current_price"), _num(t, "sma_200")
    gross = _num(c, "gross_exposure_pct")
    dd = _num(c, "drawdown_pct")
    if sent is not None and sent != 0:
        f["regime_mood"] = round(_clamp(sent) * 0.40, 3)
        notes.append(f"LENS: macro news mood {sent:+.2f} — a crude regime read (R1/R2 data feeds not wired)")
    if price and s200:
        f["asset_trend"] = 0.30 if price > s200 else -0.30
        notes.append(f"LENS: the asset's own machine is {'expanding' if price > s200 else 'contracting'} (price vs 200d)")
    if gross is not None and gross > 80:
        notes.append(f"LENS: gross exposure {gross:.0f}% — diversification, not conviction, pays for lunch")
    if dd is not None and dd > 5:
        notes.append(f"LENS: book drawdown {dd:.1f}% — respect the machine's brakes")
    return f, notes


def _lens_minervini(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    t_checks = [ch for ch in checks if ch.rule_id.startswith("T")]
    price, s50, s200 = _num(t, "current_price"), _num(t, "sma_50"), _num(t, "sma_200")
    vr = _num(t, "volume_ratio")
    sp = None
    sl = _num(t, "stop_loss")
    if price and sl and price > 0:
        sp = abs(price - sl) / price * 100.0
    if t.get("direction") == "BUY" and t_checks:
        passed = sum(1 for ch in t_checks if ch.passed)
        ratio = passed / len(t_checks)
        f["trend_template"] = round((ratio - 0.5) * 1.2, 3)
        notes.append(f"LENS: trend template {passed}/{len(t_checks)} legs pass — "
                     + ("Stage 2 confirmed" if ratio == 1.0 else "the template is incomplete"))
    elif price and s200 and price < s200 and (not s50 or s50 < s200):
        f["stage_4_decline"] = -0.50
        notes.append("LENS: below declining long MAs — Stage 4 territory, longs forbidden")
    if sp is not None:
        if sp <= 8:
            f["risk_discipline"] = 0.10 * _dirsign(t)
            notes.append(f"LENS: stop {sp:.1f}% from entry — inside the house limit")
        else:
            f["sloppy_stop"] = -0.25 * _dirsign(t)
            notes.append(f"LENS: stop {sp:.1f}% from entry — wider than I would ever allow")
    if vr is not None and vr >= 1.5 and f.get("trend_template", 0) > 0:
        f["volume_confirm"] = 0.15
        notes.append(f"LENS: {vr:.1f}x volume behind the move")
    return f, notes


def _lens_oneil(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    price, hi = _num(t, "current_price"), _num(t, "high_52w")
    vr = _num(t, "volume_ratio")
    sent = _num(t, "market_sentiment")
    score = _num(t, "score") or 0.0
    if price and hi and hi > 0:
        dist = (hi - price) / hi
        if dist <= 0.15:
            f["new_high_ground"] = 0.45
            notes.append(f"LENS: {dist:.0%} off the 52-week high — new-high ground is where big winners live (N)")
        elif dist >= 0.40:
            f["far_from_highs"] = -0.35
            notes.append(f"LENS: {dist:.0%} below the high — laggards lag for a reason")
    if vr is not None and vr >= 1.4 and score:
        f["demand_volume"] = 0.20 * (1 if score > 0 else -1)
        notes.append(f"LENS: volume {vr:.1f}x — institutional demand footprint (S)")
    if sent is not None and sent != 0:
        f["market_direction"] = 0.25 * (1 if sent > 0 else -1)
        notes.append(f"LENS: general-market mood {sent:+.2f} — three of four stocks follow the market (M)")
    return f, notes


def _lens_thorp(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    stats = c.get("module_stats") or {}
    trades = int(stats.get("trades", 0) or 0)
    ds = _dirsign(t)
    if trades < 50:
        notes.append(f"LENS: only {trades} trades of module history (need 50) — no edge estimate, I abstain")
        return f, notes
    wins = int(stats.get("wins", 0) or 0)
    p = wins / trades
    b = _num(t, "risk_reward") or 0.0
    if b <= 0:
        notes.append("LENS: undefined payoff ratio — cannot size, will not vote")
        return f, notes
    p_shrunk = 0.5 + (p - 0.5) * 0.5  # K5 estimation-error shrinkage
    f_star = (b * p_shrunk - (1 - p_shrunk)) / b
    if f_star <= 0:
        f["no_edge"] = -0.50 * ds
        notes.append(f"LENS: shrunk Kelly f*={f_star:.3f} <= 0 on module '{t.get('module')}' — a bet with no edge is a donation")
    else:
        f["kelly_edge"] = round(ds * min(0.70, f_star * 2), 3)
        notes.append(f"LENS: shrunk Kelly f*={f_star:.3f} (p={p:.2f}, b={b:.1f}) — a positive-expectation bet, sized fractionally")
        risk_frac = (_num(c, "planned_risk_pct") or 0.0) / 100.0
        cap = 0.5 * f_star
        if risk_frac > cap:
            f["oversized_bet"] = -0.30 * ds
            notes.append(f"LENS: planned risk {risk_frac:.4f} exceeds half-Kelly cap {cap:.4f} — ruin is the only unforgivable sin")
    return f, notes


def _lens_munger_bias(t: dict, c: dict, checks: list):
    f: dict[str, float] = {}
    notes: list[str] = []
    ds = _dirsign(t)
    reasons = t.get("reasons") or []
    rsi, vr = _num(t, "rsi"), _num(t, "volume_ratio")
    n = len(reasons)
    if n >= 3:
        f["lollapalooza"] = 0.35 * ds
        notes.append(f"LENS: {n} independent factors agree (M7) — a lollapalooza, not a lone anecdote")
    else:
        f["thin_thesis"] = -0.35 * ds
        notes.append(f"LENS: only {n} supporting factor(s) — one reason is a story, not a thesis (M7)")
    if rsi is not None:
        if t.get("direction") == "BUY" and rsi > 70:
            f["fomo_check"] = -0.40
            notes.append(f"LENS: buying at RSI {rsi:.0f} — social-proof tendency detected; envy is a terrible sin")
        elif t.get("direction") == "SELL" and rsi < 30:
            f["panic_check"] = 0.40
            notes.append(f"LENS: shorting at RSI {rsi:.0f} — piling onto despair is deprival-super-reaction bias")
    if vr is not None and rsi is not None and vr > 2.5 and rsi > 65:
        f["crowding"] = -0.30
        notes.append(f"LENS: {vr:.1f}x volume at RSI {rsi:.0f} — the crowd is already in the theatre (M4 proxy)")
    return f, notes


# ── persona registry ──────────────────────────────────────────────────────
# (display name, investor key in rules.yaml, lens)

PERSONAS: list[tuple[str, str, Lens]] = [
    ("Warren Buffett & Charlie Munger", "Buffett/Munger", _lens_buffett),
    ("Benjamin Graham", "Graham", _lens_graham),
    ("Jesse Livermore", "Livermore", _lens_livermore),
    ("Stanley Druckenmiller", "Druckenmiller/Soros", _lens_druckenmiller),
    ("Jim Simons", "Simons", _lens_simons),
    ("Ray Dalio", "Dalio", _lens_dalio),
    ("Mark Minervini", "Minervini", _lens_minervini),
    ("William O'Neil", "O'Neil", _lens_oneil),
    ("Edward Thorp", "Thorp", _lens_thorp),
    ("Charlie Munger (Bias Audit)", "Munger", _lens_munger_bias),
]

PERSONA_NAMES = [p[0] for p in PERSONAS]

# ── deterministic quote bank, keyed by dominant lens factor ──────────────

QUOTES: dict[str, dict[str, str]] = {
    "Warren Buffett & Charlie Munger": {
        "durable_trend": "Time is the friend of the wonderful business — and {sym} is acting like one.",
        "margin_of_safety": "Price is what you pay, value is what you get — {sym} is finally on sale.",
        "priced_for_perfection": "Be fearful when others are greedy — and they are queuing for {sym}.",
        "overpaying": "You pay a very high price for a cheery consensus.",
        "default": "Without ten years of accounts on {sym}, we mostly sit on our hands. Sitting is underrated.",
    },
    "Benjamin Graham": {
        "fear_discount": "Buy when Mr. Market is despondent — he is quoting {sym} in tears today.",
        "speculative_froth": "The speculator's chief enemy is likely to be himself — {sym} is a speculation, not an investment.",
        "near_52w_low": "The margin of safety lives near the lows, and {sym} is knocking on them.",
        "no_margin_of_safety": "An investment operation promises safety of principal — {sym} at these prices promises neither.",
        "default": "In the short run the market is a voting machine; I will wait for the weighing.",
    },
    "Jesse Livermore": {
        "line_of_least_resistance": "The trend is your friend until the end — the tape on {sym} says {dir_word}.",
        "pivotal_point": "{sym} just crossed its pivotal point; the big money is in the big swing.",
        "fresh_move": "A fresh move on {sym} — it is never your thinking that makes money, it is the sitting.",
        "tape_volume": "The tape tells the truth: volume is voting {dir_word} on {sym}.",
        "default": "There is a time to go long, a time to go short, and a time to go fishing.",
    },
    "Stanley Druckenmiller": {
        "liquidity_mood": "It's liquidity that moves markets — and it is blowing {dir_word} on {sym}.",
        "macro_trend": "Never fight the trend and never fight the Fed; {sym} is trending {dir_word}.",
        "asymmetry": "When you have conviction and asymmetry, bet big — {sym} offers both.",
        "poor_asymmetry": "The payoff on {sym} is too thin — home runs need fat pitches.",
        "default": "The first rule is capital preservation; nothing here demands a swing.",
    },
    "Jim Simons": {
        "signal_strength": "The model flags {sym} at {dir_word}-side significance; we trade the signal, not the story.",
        "volume_anomaly": "An anomaly in {sym}'s participation — patterns persist longer than reasons.",
        "default": "One signal is an anecdote. We prefer twenty — conviction stays capped.",
    },
    "Ray Dalio": {
        "regime_mood": "The machine says the regime tilts {dir_word} — position for it, hedged.",
        "asset_trend": "{sym}'s own cycle is {dir_word}; diversification does the rest.",
        "default": "He who lives by the crystal ball will eat shattered glass — stay balanced.",
    },
    "Mark Minervini": {
        "trend_template": "{sym} checks the trend template — Stage 2 is the only stage worth owning.",
        "stage_4_decline": "{sym} is in Stage 4 — hope is not a setup; there is nothing to do.",
        "sloppy_stop": "The stop on {sym} is wider than my rulebook allows — risk first, always.",
        "risk_discipline": "Risk comes first on {sym}: a tight stop and the template in your favour.",
        "volume_confirm": "Volume confirms {sym} — demand you can see is demand you can trust.",
        "default": "When conditions aren't right, the best position is no position.",
    },
    "William O'Neil": {
        "new_high_ground": "{sym} is in new-high ground — buy strength, sell weakness; it looks abnormal only to amateurs.",
        "far_from_highs": "{sym} is a laggard — the big money is made in leaders, not bargains.",
        "demand_volume": "Big volume on {sym} is the footprint of institutions — follow it {dir_word}.",
        "market_direction": "Three of four stocks follow the market, and the market leans {dir_word}.",
        "default": "The whole secret to winning is to lose the least when you're wrong.",
    },
    "Edward Thorp": {
        "kelly_edge": "The edge on {sym} is positive after shrinkage — bet a fraction of Kelly, never the whole.",
        "no_edge": "No demonstrable edge on {sym} — a wager without edge is a donation to the house.",
        "oversized_bet": "The edge is real but the stake is too large — overbetting turns winners into ruins.",
        "default": "Fewer than fifty observations is a hunch, not an edge. I pass.",
    },
    "Charlie Munger (Bias Audit)": {
        "lollapalooza": "Several forces point the same way on {sym} — a lollapalooza, and those are rare.",
        "thin_thesis": "One reason is a story, not a thesis — invert {sym}, always invert.",
        "fomo_check": "Envy is doing the buying in {sym}, not analysis — that is how clever people go broke.",
        "panic_check": "Shorting despair in {sym} is deprival-super-reaction — the crowd is already out.",
        "crowding": "Everyone is in the theatre on {sym} and the exits are narrow.",
        "default": "It is remarkable how much advantage we got by trying to be consistently not stupid.",
    },
}


def _pick_quote(persona: str, dominant: str, sym: str, dscore: float) -> str:
    bank = QUOTES.get(persona, {})
    tpl = bank.get(dominant) or bank.get("default") or ""
    return tpl.format(sym=sym.replace("/USDT", ""), dir_word="up" if dscore >= 0 else "down")


# ── persona track record (accuracy ledger, mirrors RuleStats design) ─────

class PersonaStats:
    """Per-persona win/loss ledger — the PM's evidence for vote weighting.

    A persona is "correct" on a closed trade when its vote matched the
    outcome: (agreed with trade direction AND trade won) or (disagreed AND
    trade lost). NEUTRAL votes are abstentions and never scored. Weights
    default to 1.0 until MIN_SAMPLES scored votes accumulate.
    """

    MIN_SAMPLES = 5
    WEIGHT_FLOOR = 0.5
    WEIGHT_CAP = 1.5

    def __init__(self, path: str | Path = PERSONA_STATS_PATH):
        self.path = Path(path)
        self.stats: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.stats = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning(f"Could not parse {self.path}; starting fresh persona stats")
                self.stats = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        # Write-through to Postgres (audit_events, rule_stats-style) when configured.
        store = get_store()
        if store.enabled:
            store.fire(store.log_audit_event(
                "PERSONA_STATS", None, json.loads(json.dumps(self.stats))))

    def _bucket(self, persona: str) -> dict[str, int]:
        return self.stats.setdefault(
            persona, {"wins": 0, "losses": 0, "abstains": 0})

    def record_trade_outcome(self, votes: list[dict], side: str, won: bool) -> None:
        """Score every persona's entry vote against the closed trade outcome."""
        for v in votes:
            persona = v.get("persona") if isinstance(v, dict) else v.persona
            signal = v.get("signal") if isinstance(v, dict) else v.signal
            if not persona:
                continue
            s = self._bucket(persona)
            if signal == NEUTRAL:
                s["abstains"] = s.get("abstains", 0) + 1
                continue
            agreed = (signal == BULLISH) == (side == "BUY")
            correct = agreed == won
            if correct:
                s["wins"] = s.get("wins", 0) + 1
            else:
                s["losses"] = s.get("losses", 0) + 1

    def samples(self, persona: str) -> int:
        s = self.stats.get(persona, {})
        return int(s.get("wins", 0) + s.get("losses", 0))

    def accuracy(self, persona: str) -> float | None:
        n = self.samples(persona)
        if not n:
            return None
        return self.stats[persona].get("wins", 0) / n

    def weight(self, persona: str) -> float:
        """Historical-accuracy weight in [0.5, 1.5]; 1.0 until sample floor met."""
        acc = self.accuracy(persona)
        if acc is None or self.samples(persona) < self.MIN_SAMPLES:
            return 1.0
        return round(max(self.WEIGHT_FLOOR, min(self.WEIGHT_CAP, 0.5 + acc)), 3)

    def records(self) -> list[dict]:
        """Track record per persona for the dashboard (all personas, even unscored)."""
        out = []
        for name in PERSONA_NAMES:
            s = self.stats.get(name, {})
            wins, losses = int(s.get("wins", 0)), int(s.get("losses", 0))
            n = wins + losses
            out.append({
                "persona": name,
                "wins": wins,
                "losses": losses,
                "abstains": int(s.get("abstains", 0)),
                "scored": n,
                "accuracy": round(wins / n, 3) if n else None,
                "weight": self.weight(name),
            })
        return out


# ── the engine ────────────────────────────────────────────────────────────

class PersonaEngine:
    """Runs all ten personas over one enriched setup. Deterministic only."""

    def __init__(self, brain: RuleBrain | None = None,
                 stats: PersonaStats | None = None):
        self.brain = brain or RuleBrain()
        self.stats = stats or PersonaStats()

    def evaluate(self, setup: dict, context: dict | None = None,
                 verdict: RuleVerdict | None = None) -> list[PersonaVote]:
        """Each persona votes using ONLY its own rules + its scoring lens."""
        context = context or {}
        if verdict is None:
            verdict = self.brain.evaluate_setup(setup, context)

        by_investor: dict[str, list] = {}
        for ch in verdict.checks:
            by_investor.setdefault(ch.investor, []).append(ch)

        sym = setup.get("symbol", "?")
        votes: list[PersonaVote] = []
        for display, investor, lens in PERSONAS:
            checks = by_investor.get(investor, [])
            factors, notes = lens(setup, context, checks)
            dscore = _clamp(sum(factors.values()))

            if dscore >= SIGNAL_THRESHOLD:
                signal = BULLISH
            elif dscore <= -SIGNAL_THRESHOLD:
                signal = BEARISH
            else:
                signal = NEUTRAL

            # Confidence: lens magnitude + shading from this persona's own
            # rule pass-ratio. Bounded, integer, fully reproducible.
            if not factors:
                confidence = 15
            else:
                confidence = 20.0 + 55.0 * abs(dscore)
                if checks:
                    pass_ratio = sum(1 for ch in checks if ch.passed) / len(checks)
                    confidence += (pass_ratio - 0.5) * 20.0
            confidence = int(round(max(5, min(95, confidence))))

            dominant = ""
            if factors:
                dominant = max(factors, key=lambda k: abs(factors[k]))

            reasoning = [ch.citation for ch in checks][:6] + notes
            if factors:
                reasoning.append("FACTORS: " + ", ".join(
                    f"{k} {v:+.2f}" for k, v in factors.items()))

            votes.append(PersonaVote(
                persona=display,
                investor=investor,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                quote=_pick_quote(display, dominant or "default", sym, dscore),
                dominant_factor=dominant,
            ))
        return votes

    def summary(self) -> dict:
        return {
            "personas": PERSONA_NAMES,
            "records": self.stats.records(),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
