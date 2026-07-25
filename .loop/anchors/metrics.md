<!-- FROZEN — human-owned. The optimizer must NEVER edit this file.
     Any change to this file is detected by knowledge/anchor_guard.py and
     halts the engine. Only a human may change the ground-truth metric. -->

# ANCHOR: THE GROUND-TRUTH METRIC (CONFIRMED)

**The metric the loop optimizes:**

> **Risk-adjusted return — the Sharpe ratio measured on walk-forward, OUT-OF-SAMPLE
> (unseen) periods — subject to a HARD 2% maximum-drawdown veto.**

A candidate configuration's score is its mean out-of-sample Sharpe across the
held-out splits. If the configuration breaches **2% max drawdown on the held-out
data**, its score is **`None` (an automatic fail)** — no Sharpe, however high,
can buy back a drawdown-veto breach.

## Exactly how it is measured

- **Implementation**: `backtest/engine.py`
  - `BacktestEngine.walk_forward(module, df, train_pct=0.7, n_splits=5)` — splits
    each asset's history into `n_splits` contiguous blocks and evaluates only the
    out-of-sample tail (`train_pct` in-sample, the remainder is the test window)
    of each block. In-sample data is used ONLY for parameter selection; the test
    window is never used to pick parameters.
  - `WalkForwardEvaluator.evaluate(module, datasets)` — the confirmed-metric gate.
    Holds out the most-recent `holdout_pct` (default 20%) of each asset's history
    (see `.loop/anchors/holdout.md`), runs the immutable `BacktestEngine` on that
    unseen tail, aggregates Sharpe across assets, and **returns `None` if the
    aggregate holdout max drawdown exceeds `max_drawdown_pct` (2.0%)**.

- **Sharpe definition** (`_compute_metrics`): `mean(trade returns %) /
  std(trade returns %) * sqrt(252)`. Std of 0 or <2 trades -> Sharpe 0.

- **On what data**: the OHLCV the engine actually trades — live-discovered
  crypto / US / NSE assets, on the active scan timeframe (15m intraday by
  default, or 1d in daily mode). The holdout is always the chronologically MOST
  RECENT slice, so "unseen" means "most recent, never used to tune".

## Invariants of the metric (do not weaken)
1. The score is out-of-sample Sharpe, never in-sample.
2. The 2% max-drawdown veto is a hard gate, not a penalty term.
3. Holdout data is scored but NEVER used to select parameters or weights.
