-- Bitcoin Analyser — Postgres persistence schema.
-- Applied idempotently on startup by db/store.py (CREATE ... IF NOT EXISTS).

-- COHORT MODE: engine_state holds ONE row per paper portfolio (cohort),
-- keyed by portfolio_id. A single 'DISTRIBUTED' row == the original behaviour.
CREATE TABLE IF NOT EXISTS engine_state (
    portfolio_id    TEXT NOT NULL DEFAULT 'DISTRIBUTED',
    equity          DOUBLE PRECISION NOT NULL,
    peak_equity     DOUBLE PRECISION NOT NULL,
    initial_capital DOUBLE PRECISION NOT NULL,
    target_equity   DOUBLE PRECISION NOT NULL,
    cycle           INT NOT NULL DEFAULT 0,
    params          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Migrate the old singleton shape (id INT PK CHECK id=1) idempotently: drop the
-- id PK/column, add portfolio_id (existing row defaults to DISTRIBUTED), and
-- enforce one row per cohort via a unique index (used by ON CONFLICT).
ALTER TABLE engine_state DROP CONSTRAINT IF EXISTS engine_state_pkey;
ALTER TABLE engine_state DROP COLUMN IF EXISTS id;
ALTER TABLE engine_state ADD COLUMN IF NOT EXISTS portfolio_id TEXT NOT NULL DEFAULT 'DISTRIBUTED';
CREATE UNIQUE INDEX IF NOT EXISTS engine_state_portfolio_id_key ON engine_state (portfolio_id);

-- Open positions (one row per cohort+symbol; deleted on close).
CREATE TABLE IF NOT EXISTS positions (
    portfolio_id TEXT NOT NULL DEFAULT 'DISTRIBUTED',
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    stop_loss   DOUBLE PRECISION NOT NULL,
    take_profit DOUBLE PRECISION NOT NULL,
    size        DOUBLE PRECISION NOT NULL,
    module      TEXT,
    opened_at   TEXT,
    reasons     JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE positions ADD COLUMN IF NOT EXISTS portfolio_id TEXT NOT NULL DEFAULT 'DISTRIBUTED';
-- Currency/fx/market MUST persist: an NSE position is priced in INR but the
-- book is USD. Without these columns, every restart restored a position with
-- no currency, the code defaulted it to USD/fx=1.0, and an NSE position got
-- mis-sized/mis-booked by the fx rate (~83-95x). This was the root cause of
-- the recurring "currency bug" -- corrections were made in memory but had
-- nowhere to persist, so every redeploy silently reverted them.
ALTER TABLE positions ADD COLUMN IF NOT EXISTS currency TEXT NOT NULL DEFAULT 'USD';
ALTER TABLE positions ADD COLUMN IF NOT EXISTS fx_rate DOUBLE PRECISION NOT NULL DEFAULT 1.0;
ALTER TABLE positions ADD COLUMN IF NOT EXISTS market TEXT;
ALTER TABLE positions DROP CONSTRAINT IF EXISTS positions_pkey;
CREATE UNIQUE INDEX IF NOT EXISTS positions_pid_symbol_key ON positions (portfolio_id, symbol);

-- Trade journal (OPEN rows + closed rows; upserted by (cohort, engine trade id)).
CREATE TABLE IF NOT EXISTS trades (
    id             TEXT NOT NULL,
    portfolio_id   TEXT NOT NULL DEFAULT 'DISTRIBUTED',
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol         TEXT NOT NULL,
    side           TEXT,
    entry_price    DOUBLE PRECISION,
    exit_price     DOUBLE PRECISION,
    size           DOUBLE PRECISION,
    pnl            DOUBLE PRECISION,
    reason         TEXT,
    module         TEXT,
    rule_citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    detail         JSONB NOT NULL DEFAULT '{}'::jsonb
);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS portfolio_id TEXT NOT NULL DEFAULT 'DISTRIBUTED';
-- Trade-level corrections MUST persist, exactly like the positions currency/fx
-- columns above. A restatement (restate_trade) rewrites the pnl column, but
-- without these dedicated columns the "this trade was restated" fact and the
-- pre-restatement figure lived only in the JSONB blob and were never mapped
-- back on load -- so after a restart the idempotency guard couldn't see the
-- prior restatement (double-credit risk) and the dashboard lost the marker.
-- ADD COLUMN IF NOT EXISTS backfills every existing row (restated=false).
ALTER TABLE trades ADD COLUMN IF NOT EXISTS restated BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS original_pnl DOUBLE PRECISION;
ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_pkey;
CREATE UNIQUE INDEX IF NOT EXISTS trades_pid_id_key ON trades (portfolio_id, id);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades (portfolio_id);

-- Mirror of control/decision_audit.py JSONL entries.
CREATE TABLE IF NOT EXISTS audit_events (
    id      BIGSERIAL PRIMARY KEY,
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    type    TEXT NOT NULL,
    symbol  TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_audit_events_ts ON audit_events (ts);
CREATE INDEX IF NOT EXISTS idx_audit_events_symbol ON audit_events (symbol);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events (type);

-- Per-trade P&L attribution (loops/attribution.py AttributionRecord).
CREATE TABLE IF NOT EXISTS attribution (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    trade_id   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    module     TEXT,
    regime     TEXT,
    pnl        DOUBLE PRECISION,
    ret_pct    DOUBLE PRECISION,
    components JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_attribution_ts ON attribution (ts);
CREATE INDEX IF NOT EXISTS idx_attribution_symbol ON attribution (symbol);
CREATE INDEX IF NOT EXISTS idx_attribution_module_regime ON attribution (module, regime);

-- Per-rule win/loss ledger (knowledge/rules.py RuleStats).
CREATE TABLE IF NOT EXISTS rule_stats (
    rule_id    TEXT PRIMARY KEY,
    wins       INT NOT NULL DEFAULT 0,
    losses     INT NOT NULL DEFAULT 0,
    weight     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    data       JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Engine memory log (the dashboard's narrative feed).
CREATE TABLE IF NOT EXISTS memory_log (
    id   BIGSERIAL PRIMARY KEY,
    ts   TIMESTAMPTZ NOT NULL DEFAULT now(),
    type TEXT NOT NULL DEFAULT 'INFO',
    msg  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_log_ts ON memory_log (ts);

-- Equity curve, one point per cohort per cycle.
CREATE TABLE IF NOT EXISTS equity_curve (
    id           BIGSERIAL PRIMARY KEY,
    portfolio_id TEXT NOT NULL DEFAULT 'DISTRIBUTED',
    ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    equity       DOUBLE PRECISION NOT NULL,
    drawdown_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    positions    INT NOT NULL DEFAULT 0
);
ALTER TABLE equity_curve ADD COLUMN IF NOT EXISTS portfolio_id TEXT NOT NULL DEFAULT 'DISTRIBUTED';
CREATE INDEX IF NOT EXISTS idx_equity_curve_ts ON equity_curve (ts);
CREATE INDEX IF NOT EXISTS idx_equity_curve_portfolio ON equity_curve (portfolio_id);
