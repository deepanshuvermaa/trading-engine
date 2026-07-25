<!-- FROZEN — human-owned. The optimizer must NEVER edit this file.
     Any change is detected by knowledge/anchor_guard.py and halts the engine. -->

# ANCHOR: THE VETO INVARIANTS (NEVER WEAKEN)

These are risk-engine vetoes. The learning loop may NEVER relax, disable,
down-weight below 1.0, re-parameterize, or search around them. They are pinned
at weight 1.0 in `knowledge/rules.py::RuleStats.weight` and excluded from
`underperformers()` by construction.

## 1. The 19 IMMUTABLE investor rules (source: `knowledge/rules.yaml`)

| ID   | Investor              | Invariant |
|------|-----------------------|-----------|
| M3   | Munger                | Cost-basis blindness — entry price is not an input to hold/exit logic |
| M4   | Munger                | Crowding filter — reject squeeze/over-owned/over-funded crowded longs |
| G4   | Graham                | Net-net diversification floor — >=30 names, single name <=3.3% |
| L9   | Livermore             | Never average down into a losing position |
| L10  | Livermore             | 10% hard bucket-shop stop — max single-trade loss ceiling |
| D4   | Druckenmiller/Soros   | Asymmetry filter — macro sleeve reward:risk >= 3:1 |
| D8   | Druckenmiller/Soros   | Drawdown circuit breaker — -5% cut gross, -10% flat + cooloff |
| D9   | Druckenmiller/Soros   | House-money gross caps (base 100% / +10% -> 150% / <0 -> 70%) |
| S4   | Simons                | Never override the model except kill-switch (flatten) |
| R10  | Dalio                 | Correlation-shock brake — avg corr >0.8 cut leverage to 0.5x |
| V7   | Minervini             | Initial stop ceiling — never > 10% below entry |
| V8   | Minervini             | Risk per trade <= 1.25% of equity |
| V9   | Minervini             | Position size cap — max single position 20-25% of book |
| V10  | Minervini             | Reward:risk at entry >= 2:1 |
| V12  | Minervini             | Win-rate throttle — <40% halve, <30% stop and paper-trade |
| V13  | Minervini             | Progressive exposure — re-enter at 25-50% size after drawdown |
| CH10 | O'Neil                | Hard 7-8% stop below purchase (breakout sleeve) |
| K4   | Thorp                 | Fractional-Kelly cap — never bet more than 0.5x Kelly |
| K7   | Thorp                 | Per-position risk cap <= 2% of equity (stat-arb) |

`knowledge/brain.py::RuleBrain.evaluate_setup` treats any FAILED immutable rule
as a hard veto (`vetoed=True`), which rejects the trade outright.

## 2. Portfolio-level hard gates (also frozen)

- **2% MAX DRAWDOWN** — `config/loader.py::RiskConfig.max_drawdown_pct = 2.0`,
  enforced in `risk/engine.py` and as the walk-forward metric veto
  (`.loop/anchors/metrics.md`). The loop may never raise this.
- **Position-size caps** — `RiskConfig.max_position_pct` and `V9`/`V8`/`K7`
  above; the loop may never raise the notional or per-trade risk caps.
- **R:R >= 2:1** — `V10` (all sleeves) and `D4` (>=3:1 macro sleeve); the entry
  reward:risk floor may never be lowered.

## 3. Enforcement
- `knowledge/anchor_guard.py::assert_anchors_untouched()` hashes the three anchor
  files and halts the engine if any changed.
- The daily loop calls `assert_anchors_untouched()` BEFORE doing anything, and
  only ever adjusts LEARNABLE rule weights — never anything on this list.
