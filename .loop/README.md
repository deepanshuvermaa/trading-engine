# `.loop/` — the self-improvement substrate (6 layers)

This directory is the Karpathy keep-or-revert loop's memory and its human-owned
guardrails. The engine reads it at startup and the daily outer loop writes to it.

## The 6 layers

1. **`PROJECT_MODEL.md`** — the map. What the system is, its entry points, module
   dependency edges, where measurable signals live, the confirmed metric, and the
   honest gaps. Human-maintained; read it first.

2. **`anchors/`** — FROZEN, human-owned ground truth. The optimizer must NEVER
   edit these; `knowledge/anchor_guard.py` hashes them and halts the engine if any
   changed.
   - `metrics.md` — the confirmed metric: out-of-sample walk-forward Sharpe with a
     hard 2% max-drawdown veto, and exactly how it is measured.
   - `holdout.md` — the held-out scheme: last 20% of each asset's history, scored
     but never optimized against.
   - `veto.md` — the invariants that may never be weakened: the 19 immutable rules,
     2% max drawdown, position-size caps, R:R >= 2:1.

3. **`experiments/`** — one JSON record per loop cycle:
   `{id, timestamp, hypothesis, diff_summary, metric_before, metric_after,
   holdout_score, verdict, verifier_notes}`. Verdict ∈ KEEP | REVERT | VETO | SEED.

4. **`lessons.md`** — compressed, append-only "tried X -> regressed Y -> don't
   retry". The loop consults it before re-proposing a hypothesis so it never
   re-walks a known dead end.

5. **`state.json`** — the loop's live memory: `best_known_config`, `best_metric`
   (mean OOS holdout Sharpe of the best config), `best_holdout_sharpe`,
   `last_outer_run`, `open_hypotheses[]`, `anchors_frozen`.

6. **`RUN.md`** — the exact commands to trigger one loop cycle and one fan-out.

## The loop, in one sentence
Each day: verify the anchors are frozen -> read LIVE closed-trade attribution +
rule_stats -> score the current config against the held-out walk-forward Sharpe
(2% DD veto) -> down-weight only LEARNABLE rules that hurt it -> KEEP or REVERT ->
write an experiment record + update `state.json` and `lessons.md`.

## Cardinal rule
The loop edits LEARNABLE things (rule weights, search bias, tunable params).
It never touches anything under `anchors/`. If it needs to, a human decides.
