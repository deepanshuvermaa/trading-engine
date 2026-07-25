# Deploying to Railway ($5 Hobby plan)

The engine is stateless-process / stateful-database: all durable state
(equity, open positions, trade journal, rule ledgers, attribution, memory
log, equity curve, audit events) lives in Postgres via `db/store.py`. On boot
the engine reconnects and resumes exactly where it left off. Without
`DATABASE_URL` it silently falls back to the local JSONL/file behaviour, so
local dev is unchanged.

## Memory budget

FinBERT + torch need ~1.5 GB RAM. On the $5 plan you have two options:

| Mode | Env | RAM | Effect |
|------|-----|-----|--------|
| Full sentiment | (default) | ~1.5–2 GB | FinBERT scores every headline |
| Lean | `SKIP_FINBERT=1` | ~400–600 MB | Neutral sentiment, torch never loaded |

If the service OOMs or restarts under memory pressure, set `SKIP_FINBERT=1`.
Everything else (scanning, rules, risk, attribution, dashboard) is unaffected.

## 1. Create the project

### Via GitHub (recommended)

1. Push this repo to GitHub (`git push`).
2. In Railway: **New Project → Deploy from GitHub repo** → pick the repo.
3. Railway detects the `Dockerfile` automatically (`railway.toml` pins it).

### Via CLI

```bash
npm i -g @railway/cli    # or: brew install railway
railway login
railway init             # from the repo root
railway up               # builds the Dockerfile and deploys
```

## 2. Add the Postgres addon

1. In the project canvas: **+ New → Database → Add PostgreSQL**.
2. Railway automatically injects `DATABASE_URL` into every service in the
   project — no manual wiring needed. (Verify under your service →
   **Variables**: you should see `DATABASE_URL` referenced from Postgres.
   If not, add it manually as `${{Postgres.DATABASE_URL}}`.)
3. The schema is applied automatically on first boot
   (`db/schema.sql`, idempotent `CREATE TABLE IF NOT EXISTS`).

## 3. Set environment variables

Service → **Variables** → add:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SKIP_FINBERT` | unset | `1` = neutral sentiment, no torch load (recommended on $5 plan) |
| `INITIAL_CAPITAL` | `100` | Starting paper capital (USD) |
| `TARGET_EQUITY` | `120` | Engine stops when equity reaches this |
| `SCAN_INTERVAL_MINUTES` | `30` | Rescan cadence |
| `PORT` | `8050` | Injected by Railway automatically — do not set manually |

Note: once the engine has saved state to Postgres, `INITIAL_CAPITAL` /
`TARGET_EQUITY` from the DB win on restart (the engine resumes where it left
off). Change them live from the dashboard's Editor's Desk, or reset the
`engine_state` table to start over:

```sql
TRUNCATE engine_state, positions, trades, equity_curve, memory_log;
-- keep rule_stats + attribution if you want the brain to retain its learning
```

## 4. Expose the dashboard

Service → **Settings → Networking → Generate Domain**. Railway routes the
domain to the injected `PORT`, which `autonomous.py` reads (default 8050
locally). The dashboard, WebSocket feed, and REST API are all on that port.
Health check is `GET /api/state` (configured in `railway.toml`).

## 5. Verify

- Logs should show: `Postgres store initialised (schema applied)` then either
  `RESUMED from Postgres: cycle N, equity $...` or
  `Persistence: Postgres connected — fresh state, first snapshot saved`.
- With `SKIP_FINBERT=1` you should NOT see `Loading sentiment model:
  ProsusAI/finbert` in logs.
- Kill/redeploy the service: equity, open positions, journal, rule weights,
  and the memory log must survive the restart.

## Local development

Nothing changes. Without `DATABASE_URL` the store is a no-op and the engine
uses the existing files (`reports/audit/*.jsonl`,
`knowledge/rule_stats.json`). To test persistence locally:

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=dev -p 5432:5432 postgres:16
set DATABASE_URL=postgresql://postgres:dev@localhost:5432/postgres   # Windows
python autonomous.py
```
