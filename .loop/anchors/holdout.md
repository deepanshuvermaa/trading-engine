<!-- FROZEN — human-owned. The optimizer must NEVER edit this file.
     Any change is detected by knowledge/anchor_guard.py and halts the engine. -->

# ANCHOR: THE HELD-OUT WALK-FORWARD SCHEME

The loop MAY be **scored** on the most-recent unseen period. The loop must
**NEVER optimize against it** — no parameter, weight, threshold, or rule
down-weight may be selected because it improved the holdout score.

## The split (frozen)

- **Holdout = the chronologically LAST 20% of each asset's available history.**
  - `holdout_pct = 0.20`.
  - Applied per asset independently (each symbol's own last 20% of bars).
  - The first 80% is the **development** region: parameter search, rule-weight
    learning, and inner-loop experiments may use ONLY this region.
  - The last 20% is the **holdout** region: used ONLY to score/verify a config
    that was already fixed on the development region.

- **Minimum data**: an asset needs >= 60 bars total and >= 30 holdout bars to be
  scored; assets below that are skipped (they do not contribute to the metric).

- **Recency**: because the split is by time and the holdout is the tail, the
  score always reflects the freshest, never-tuned-on market behaviour — the
  honest proxy for "would this have worked going forward".

## Rules of use
1. Parameter selection, rule-weight updates, and hypothesis acceptance read the
   development region only.
2. The holdout region is read exactly once per evaluation, to produce a score.
3. A config is KEPT only if it does not regress the holdout metric AND does not
   breach the 2% drawdown veto on holdout (see `.loop/anchors/metrics.md`).
4. If a change improves development but regresses holdout, it is REVERTED and the
   lesson recorded in `.loop/lessons.md` (classic overfit signature).
