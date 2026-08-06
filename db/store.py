"""Postgres persistence layer — asyncpg-backed, no-op without DATABASE_URL.

Design:
- `Store` is a thin async wrapper over an asyncpg pool. Every public method is
  safe to call whether or not Postgres is configured: with no DATABASE_URL (or
  asyncpg missing, or the connection failing) the store stays disabled and all
  calls are no-ops, so local dev keeps the existing JSONL/file behaviour.
- Synchronous call sites (DecisionAudit._persist, RuleStats.save,
  TradeAttribution._persist) write through via `Store.fire(coro)`, which
  schedules a fire-and-forget task on the running event loop. JSONL files are
  still written first — the DB is a mirror, never a gate on the trading loop.
- `get_store()` returns the process-wide singleton so modules can share one
  pool without wiring changes.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from utils.logger import get_logger

try:
    import asyncpg  # type: ignore
except ImportError:  # pragma: no cover — local dev without asyncpg
    asyncpg = None

log = get_logger("db.store")

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _jsonable(obj: Any) -> Any:
    """Deep-copy to plain JSON types (detaches from live mutable state)."""
    return json.loads(json.dumps(obj, default=str))


class Store:
    """Async Postgres store. Disabled (all no-ops) unless init() connects."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn if dsn is not None else os.environ.get("DATABASE_URL")
        self.pool = None
        self._tasks: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return self.pool is not None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def init(self) -> "Store":
        """Connect and apply schema. Never raises — falls back to disabled."""
        if not self.dsn:
            log.info("DATABASE_URL not set — Postgres persistence disabled "
                     "(local JSONL/file fallback active)")
            return self
        if asyncpg is None:
            log.warning("asyncpg not installed — Postgres persistence disabled")
            return self
        try:
            async def _init_conn(conn):
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads,
                    schema="pg_catalog")

            # Small pool — Railway $5 plan is memory-constrained.
            self.pool = await asyncpg.create_pool(
                self.dsn, min_size=1, max_size=3, command_timeout=30,
                init=_init_conn)
            async with self.pool.acquire() as conn:
                await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            log.info("Postgres store initialised (schema applied)")
        except Exception as e:
            log.error(f"Postgres init failed — persistence disabled: {e}")
            self.pool = None
        return self

    async def close(self) -> None:
        if self.pool is not None:
            for t in list(self._tasks):
                if not t.done():
                    try:
                        await asyncio.wait_for(t, timeout=5)
                    except Exception:
                        pass
            await self.pool.close()
            self.pool = None

    # ── fire-and-forget bridge for sync call sites ───────────────────────

    def fire(self, coro) -> None:
        """Schedule a store coroutine without awaiting it. No-op if disabled
        or if there is no running event loop (e.g. sync scripts/tests)."""
        if not self.enabled:
            coro.close()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            return
        task = loop.create_task(self._guard(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guard(self, coro) -> None:
        try:
            await coro
        except Exception as e:
            log.warning(f"DB write-through failed (JSONL fallback intact): {e}")

    async def _execute(self, sql: str, *args) -> None:
        if not self.enabled:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(sql, *args)

    async def _fetch(self, sql: str, *args):
        if not self.enabled:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, *args)

    # ── clean-slate reset ────────────────────────────────────────────────

    # Only these tables may ever be wiped by the admin reset endpoint. A hard
    # whitelist because the table name can't be a bound parameter (it's
    # interpolated into the TRUNCATE), so this is the SQL-injection guard.
    RESETTABLE_TABLES = frozenset({
        "trades", "positions", "equity_curve", "memory_log",
        "audit_events", "attribution", "rule_stats",
    })

    async def truncate_tables(self, tables) -> list[str]:
        """TRUNCATE each whitelisted table. Returns the tables actually wiped.
        No-op (returns []) when persistence is disabled."""
        if not self.enabled:
            return []
        done: list[str] = []
        async with self.pool.acquire() as conn:
            for t in tables:
                if t not in self.RESETTABLE_TABLES:
                    log.warning(f"truncate_tables: refusing unknown table {t!r}")
                    continue
                await conn.execute(f"TRUNCATE TABLE {t} RESTART IDENTITY")
                done.append(t)
        return done

    # ── engine_state (one row PER cohort, keyed by portfolio_id) ──────────

    async def save_engine_state(self, state: dict) -> None:
        await self._execute(
            """
            INSERT INTO engine_state
                (portfolio_id, equity, peak_equity, initial_capital,
                 target_equity, cycle, params, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, now())
            ON CONFLICT (portfolio_id) DO UPDATE SET
                equity = EXCLUDED.equity,
                peak_equity = EXCLUDED.peak_equity,
                initial_capital = EXCLUDED.initial_capital,
                target_equity = EXCLUDED.target_equity,
                cycle = EXCLUDED.cycle,
                params = EXCLUDED.params,
                updated_at = now()
            """,
            state.get("portfolio_id") or "DISTRIBUTED",
            float(state["equity"]), float(state["peak_equity"]),
            float(state["initial_capital"]), float(state["target_equity"]),
            int(state.get("cycle", 0)), _jsonable(state.get("params") or {}),
        )

    async def load_engine_states(self) -> list[dict]:
        """Every cohort's persisted state row."""
        rows = await self._fetch("SELECT * FROM engine_state")
        return [
            {
                "portfolio_id": r["portfolio_id"],
                "equity": r["equity"],
                "peak_equity": r["peak_equity"],
                "initial_capital": r["initial_capital"],
                "target_equity": r["target_equity"],
                "cycle": r["cycle"],
                "params": r["params"] or {},
            }
            for r in rows
        ]

    async def load_engine_state(self) -> dict | None:
        """Backward-compat single-row loader (returns the DISTRIBUTED cohort,
        else the first row)."""
        rows = await self.load_engine_states()
        if not rows:
            return None
        for r in rows:
            if r["portfolio_id"] == "DISTRIBUTED":
                return r
        return rows[0]

    # ── positions ────────────────────────────────────────────────────────

    async def upsert_position(self, p: dict) -> None:
        await self._execute(
            """
            INSERT INTO positions
                (portfolio_id, symbol, side, entry_price, stop_loss,
                 take_profit, size, module, opened_at, reasons,
                 currency, fx_rate, market, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, now())
            ON CONFLICT (portfolio_id, symbol) DO UPDATE SET
                side = EXCLUDED.side,
                entry_price = EXCLUDED.entry_price,
                stop_loss = EXCLUDED.stop_loss,
                take_profit = EXCLUDED.take_profit,
                size = EXCLUDED.size,
                module = EXCLUDED.module,
                opened_at = EXCLUDED.opened_at,
                reasons = EXCLUDED.reasons,
                currency = EXCLUDED.currency,
                fx_rate = EXCLUDED.fx_rate,
                market = EXCLUDED.market,
                updated_at = now()
            """,
            p.get("portfolio_id") or "DISTRIBUTED",
            p["symbol"], p["side"], float(p["entry_price"]),
            float(p["stop_loss"]), float(p["take_profit"]), float(p["size"]),
            p.get("module"), p.get("opened_at"),
            _jsonable(p.get("reasons") or []),
            p.get("currency") or "USD", float(p.get("fx_rate") or 1.0),
            p.get("market"),
        )

    async def delete_position(self, symbol: str,
                              portfolio_id: str = "DISTRIBUTED") -> None:
        await self._execute(
            "DELETE FROM positions WHERE portfolio_id = $1 AND symbol = $2",
            portfolio_id, symbol)

    async def load_positions(self) -> list[dict]:
        rows = await self._fetch("SELECT * FROM positions ORDER BY opened_at")
        return [
            {
                "portfolio_id": r["portfolio_id"],
                "symbol": r["symbol"], "side": r["side"],
                "entry_price": r["entry_price"], "stop_loss": r["stop_loss"],
                "take_profit": r["take_profit"], "size": r["size"],
                "module": r["module"], "opened_at": r["opened_at"],
                "reasons": r["reasons"] or [],
                # Restore currency/fx/market so an NSE position comes back as
                # INR/its-fx, not the USD default -- the field that was missing
                # end-to-end (schema, upsert, and here) and caused the
                # recurring currency reversion on every restart. The schema's
                # ALTER ... ADD COLUMN IF NOT EXISTS backfills these on every
                # existing row, so they're always present after init().
                "currency": r["currency"] or "USD",
                "fx_rate": r["fx_rate"] or 1.0,
                "market": r["market"],
            }
            for r in rows
        ]

    # ── trades ───────────────────────────────────────────────────────────

    async def save_trade(self, t: dict) -> None:
        await self._execute(
            """
            INSERT INTO trades
                (id, portfolio_id, symbol, side, entry_price, exit_price, size,
                 pnl, reason, module, rule_citations, detail,
                 restated, original_pnl)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (portfolio_id, id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                side = EXCLUDED.side,
                entry_price = EXCLUDED.entry_price,
                exit_price = EXCLUDED.exit_price,
                size = EXCLUDED.size,
                pnl = EXCLUDED.pnl,
                reason = EXCLUDED.reason,
                module = EXCLUDED.module,
                rule_citations = EXCLUDED.rule_citations,
                detail = EXCLUDED.detail,
                restated = EXCLUDED.restated,
                original_pnl = EXCLUDED.original_pnl
            """,
            t["id"], t.get("portfolio_id") or "DISTRIBUTED",
            t["symbol"], t.get("side"),
            _f(t.get("entry")), _f(t.get("exit")), _f(t.get("size")),
            _f(t.get("pnl")), t.get("reason"), t.get("module"),
            _jsonable(t.get("rule_citations") or []),
            _jsonable({k: v for k, v in t.items()
                       if k not in ("rule_citations",)}),
            bool(t.get("restated", False)), _f(t.get("original_pnl")),
        )

    async def load_trades(self, limit: int = 1000,
                          portfolio_id: str = "DISTRIBUTED") -> list[dict]:
        rows = await self._fetch(
            "SELECT * FROM (SELECT * FROM trades WHERE portfolio_id = $2 "
            "ORDER BY id DESC LIMIT $1) s ORDER BY id ASC", limit, portfolio_id)
        out = []
        for r in rows:
            d = r["detail"] or {}
            out.append({
                "id": r["id"],
                "portfolio_id": r["portfolio_id"],
                "date": d.get("date"),
                "symbol": r["symbol"], "side": r["side"],
                "entry": r["entry_price"], "exit": r["exit_price"],
                "size": r["size"], "pnl": r["pnl"],
                "reason": r["reason"], "module": r["module"],
                # Restatement audit fields — mapped back so the idempotency
                # guard in restate_trade survives a restart and the dashboard
                # keeps the "restated" marker. pnl already holds the restated
                # value (save_trade rewrote it); original_pnl is the pre-fix
                # figure. restatement_reason lives in the detail JSONB blob.
                "restated": bool(r["restated"]),
                "original_pnl": r["original_pnl"],
                "restatement_reason": d.get("restatement_reason"),
            })
        return out

    # ── audit events ─────────────────────────────────────────────────────

    async def log_audit_event(self, event_type: str, symbol: str | None,
                              payload: dict) -> None:
        await self._execute(
            "INSERT INTO audit_events (type, symbol, payload) "
            "VALUES ($1, $2, $3)",
            event_type or "UNKNOWN", symbol, _jsonable(payload))

    # ── attribution ──────────────────────────────────────────────────────

    async def save_attribution(self, rec: dict) -> None:
        await self._execute(
            """
            INSERT INTO attribution
                (trade_id, symbol, module, regime, pnl, ret_pct, components)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            rec.get("trade_id", ""), rec.get("symbol", ""),
            rec.get("module"), rec.get("regime"),
            _f(rec.get("pnl")), _f(rec.get("ret_pct")), _jsonable(rec))

    async def load_regime_perf(self) -> dict[str, dict[str, float]]:
        """Rebuild (module|regime) -> {sum_ret, n} from the attribution table."""
        rows = await self._fetch(
            "SELECT module, regime, COALESCE(SUM(ret_pct), 0) AS sum_ret, "
            "COUNT(*) AS n FROM attribution GROUP BY module, regime")
        return {
            f"{r['module'] or 'unknown'}|{r['regime'] or 'unknown'}":
                {"sum_ret": float(r["sum_ret"]), "n": int(r["n"])}
            for r in rows
        }

    # ── rule stats ───────────────────────────────────────────────────────

    async def save_rule_stats(self, stats: dict[str, dict]) -> None:
        if not self.enabled or not stats:
            return
        rows = [
            (rule_id, int(s.get("wins", 0)), int(s.get("losses", 0)),
             float(s.get("weight", 1.0)), _jsonable(s))
            for rule_id, s in stats.items()
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO rule_stats (rule_id, wins, losses, weight, data,
                                        updated_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (rule_id) DO UPDATE SET
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    weight = EXCLUDED.weight,
                    data = EXCLUDED.data,
                    updated_at = now()
                """, rows)

    async def load_rule_stats(self) -> dict[str, dict] | None:
        rows = await self._fetch("SELECT rule_id, data FROM rule_stats")
        if not rows:
            return None
        return {r["rule_id"]: (r["data"] or {}) for r in rows}

    # ── memory log ───────────────────────────────────────────────────────

    async def log_memory(self, entry: dict) -> None:
        await self._execute(
            "INSERT INTO memory_log (type, msg) VALUES ($1, $2)",
            entry.get("type", "INFO"), entry.get("msg", ""))

    async def load_memory(self, limit: int = 100) -> list[dict]:
        rows = await self._fetch(
            "SELECT * FROM (SELECT * FROM memory_log ORDER BY id DESC "
            "LIMIT $1) s ORDER BY id ASC", limit)
        return [
            {"ts": r["ts"].isoformat(), "msg": r["msg"], "type": r["type"]}
            for r in rows
        ]

    # ── equity curve ─────────────────────────────────────────────────────

    async def log_equity_point(self, pt: dict,
                               portfolio_id: str = "DISTRIBUTED") -> None:
        await self._execute(
            "INSERT INTO equity_curve (portfolio_id, equity, drawdown_pct, "
            "positions) VALUES ($1, $2, $3, $4)",
            portfolio_id,
            float(pt.get("equity", 0.0)), float(pt.get("drawdown_pct", 0.0)),
            int(pt.get("positions", 0)))

    async def load_equity_curve(self, limit: int = 500,
                                portfolio_id: str = "DISTRIBUTED") -> list[dict]:
        rows = await self._fetch(
            "SELECT * FROM (SELECT * FROM equity_curve WHERE portfolio_id = $2 "
            "ORDER BY id DESC LIMIT $1) s ORDER BY id ASC", limit, portfolio_id)
        return [
            {
                "date": r["ts"].isoformat(), "equity": r["equity"],
                "drawdown_pct": r["drawdown_pct"], "positions": r["positions"],
            }
            for r in rows
        ]


def _f(v) -> float | None:
    return float(v) if v is not None else None


# ── process-wide singleton ───────────────────────────────────────────────

_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
