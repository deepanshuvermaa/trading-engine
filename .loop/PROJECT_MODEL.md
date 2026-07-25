# PROJECT MODEL — Bitcoin Analyser / Multi-Asset Macro Trading Engine

**What this is.** A deterministic, self-improving paper-trading engine that scans a
live-discovered universe of crypto, US equities and NSE equities, scores setups
with technical + structural signals, filters every candidate through a codified
registry of 19 immutable + ~140 learnable investor rules (Buffett, Graham,
Livermore, Druckenmiller, Simons, Dalio, Minervini, O'Neil, Thorp) plus a
deterministic 10-persona "Partners' Room", sizes and books trades in TRUE USD
(with a per-market cost model and INR->USD FX), protects open positions with a
fast Sentry loop, and runs a daily Karpathy keep-or-revert loop that scores the
current configuration against a walk-forward, out-of-sample Sharpe metric with a
hard 2% max-drawdown veto. **No LLM is anywhere in the trading loop.**

## Entry points
- `autonomous.py` — the live engine. Scout loop (universe scan -> score -> rules
  -> personas -> open), Sentry loop (mark/trail/close open positions), the daily
  outer learning loop, and the FastAPI dashboard, all run concurrently.
- `backtest/engine.py` — the immutable verifier. `BacktestEngine.walk_forward`
  and `WalkForwardEvaluator` produce the ground-truth out-of-sample metric.

## Core modules + dependency edges
```
autonomous.AutonomousEngine
  ├─ data/universe.py            (dynamic symbol discovery: US/India/crypto)
  ├─ data/ingestion/*.py         (crypto, us_equity, indian_equity OHLCV)
  ├─ data/fx.py                  (USDINR rate, cached 6h, fallback 83.0)   [Task 1]
  ├─ data/costs.py               (per-market round-trip cost model)        [Task 1]
  ├─ indicators/{technical,structural}.py
  ├─ knowledge/brain.py          (RuleBrain: deterministic rule evaluation)
  │    └─ knowledge/rules.py      (RuleBook + RuleStats weights)
  │         └─ knowledge/rules.yaml (19 IMMUTABLE + LEARNABLE rules)
  ├─ knowledge/anchor_guard.py   (freeze-checks the .loop anchors)         [Task 2]
  ├─ personas/{engine,manager}.py (10 deterministic investor personas)
  ├─ loops/attribution.py        (post-trade P&L decomposition -> rule_stats)
  ├─ macro/intelligence.py       (news/sentiment, 30-min cached)
  ├─ risk/engine.py              (absolute-veto risk engine; 2% max DD)
  └─ dashboard/{server,index.html} (newspaper-style control room)

backtest/engine.py  (verifier)
  ├─ WalkForwardEvaluator        (confirmed metric: OOS Sharpe + 2% DD veto)  [Task 3]
  ├─ strategy/modules/*.py       (trend_follower, mean_reverter, breakout)
  └─ risk/engine.py

loops/outer_loop.py  ← reads knowledge/rule_stats.json (LIVE attribution)   [Task 3]
loops/inner_loop.py  ← backtest experiment search (offline)
```

## Where the measurable signals live
- **Backtest metrics**: `backtest/engine.py::_compute_metrics` -> `BacktestResult`
  (sharpe, sortino, max_drawdown, win_rate, profit_factor, calmar). Out-of-sample
  splits: `BacktestEngine.walk_forward` and `WalkForwardEvaluator.evaluate`.
- **Live attribution**: `loops/attribution.py` decomposes each closed trade
  (thesis / timing / sizing / exit-efficiency / regime-penalty) and updates
  `knowledge/rule_stats.json` per-rule win/loss ledgers.
- **Live P&L / cost**: `autonomous.py` — TRUE-USD equity, `cost_drag_total`,
  per-position `est_round_trip_cost_*`.

## The confirmed optimization metric
**Risk-adjusted return (Sharpe) on walk-forward UNSEEN periods, with the 2%
max-drawdown veto as a hard gate.** Defined in `.loop/anchors/metrics.md`;
measured by `WalkForwardEvaluator` on the held-out split defined in
`.loop/anchors/holdout.md`. A config that breaches 2% DD on holdout scores
`None` (fail) regardless of its Sharpe.

## Honest gaps
- Fundamental rules (Buffett/Graham/O'Neil fundamentals) lack data feeds, so most
  are reported not-applicable and never fire — only ~25 rules have live evaluators.
- Attribution's `sizing_error` / `exit_efficiency_atr` terms use the setup's
  native ATR/stop; for INR assets these two secondary terms mix units (headline
  thesis/timing P&L and all realized/unrealized P&L are correct TRUE USD).
- WalkForward holdout is scored on whatever OHLCV the last live scan fetched
  (intraday 15m by default); it is a proxy, not a full research-grade backtest.
- The daily loop derives rule down-weighting from live `rule_stats` accuracy; it
  does not mutate learnable thresholds directly (that path stays in loops/inner_loop).
