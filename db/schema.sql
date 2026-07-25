-- Bitcoin Analyser — Postgres persistence schema.
-- Applied idempotently on startup by db/store.py (CREATE ... IF NOT EXISTS).

-- Singleton engine state: one row, id pinned to 1.
CREATE TABLE IF NOT EXISTS engine_state (
    id              INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    equity          DOUBLE PRECISION NOT NULL,
    peak_equity     DOUBLE PRECISION NOT NULL,
    initial_capital DOUBLE PRECISION NOT NULL,
    target_equity   DOUBLE PRECISION NOT NULL,
    cycle           INT NOT NULL DEFAULT 0,
    params          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Open positions (one row per symbol; deleted on close).
CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
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

-- Trade journal (OPEN rows + closed rows; upserted by engine trade id).
CREATE TABLE IF NOT EXISTS trades (
    id             TEXT PRIMARY KEY,
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
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);

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

-- Equity curve, one point per cycle.
CREATE TABLE IF NOT EXISTS equity_curve (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    equity       DOUBLE PRECISION NOT NULL,
    drawdown_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    positions    INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_equity_curve_ts ON equity_curve (ts);
