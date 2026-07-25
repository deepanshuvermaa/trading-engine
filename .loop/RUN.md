# RUN — trigger a loop cycle and a fan-out

All commands run from the repo root with the project venv
(`./venv/Scripts/python.exe` on Windows).

## 0. Verify the anchors are frozen (always first)
```bash
./venv/Scripts/python.exe -c "from knowledge.anchor_guard import assert_anchors_untouched; assert_anchors_untouched(); print('Anchors verified frozen.')"
```
The engine also does this automatically at startup and before every daily loop.

## 1. One loop cycle (daily keep-or-revert against the confirmed metric)
The live engine runs the cycle on a 24h wall-clock cadence automatically. To
force one cycle immediately against the current live state:
```bash
./venv/Scripts/python.exe -c "import asyncio, autonomous; \
e = autonomous.AutonomousEngine(); \
asyncio.run(e.run_daily_outer_loop(force=True))"
```
This: verifies anchors -> reads live attribution + rule_stats -> scores the
current config on the held-out walk-forward Sharpe (2% DD veto) -> down-weights
weak LEARNABLE rules -> writes a record to `.loop/experiments/` -> updates
`.loop/state.json` and `.loop/lessons.md`.

## 2. Score the current config on the held-out metric only (no writes)
```bash
./venv/Scripts/python.exe -c "import asyncio, autonomous; \
e = autonomous.AutonomousEngine(); \
print(asyncio.run(e.score_holdout_metric()))"
```

## 3. One fan-out (offline inner-loop parameter search across modules x assets)
The bilevel search that proposes/keeps parameter changes on the DEVELOPMENT
region only (never the holdout):
```bash
./venv/Scripts/python.exe -m control.overseer  # full fetch -> inner loop -> daily review
```
or programmatically:
```bash
./venv/Scripts/python.exe -c "import asyncio; from control.overseer import Overseer; \
o = Overseer(); asyncio.run(o.run_full_cycle())"
```

## 4. Launch the live engine (dashboard on :8050)
```bash
./venv/Scripts/python.exe autonomous.py
# then open http://localhost:8050  (THE IMPROVEMENT LOOP panel shows loop status)
```
