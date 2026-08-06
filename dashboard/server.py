"""Dashboard API — real-time WebSocket + REST for the trading control room."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from utils.logger import get_recent_logs

app = FastAPI(title="Trading Engine Control Room")

# Live engine reference — set by autonomous.start_engine() so the control
# endpoints below can mutate the running AutonomousEngine instance.
ENGINE: Any = None

# Global state — set by the autonomous loop
ENGINE_STATE: dict[str, Any] = {
    "status": "IDLE",
    "capital": 100.0,
    "target": 120.0,
    "scan_interval_minutes": 30,
    "max_positions": 3,
    "max_position_pct": 10.0,
    "scan_timeframe": "15m",
    "sentry_interval_seconds": 45,
    "trail_activation_atr": 1.5,
    "trail_distance_atr": 1.0,
    "last_sentry_run": None,
    "paused": False,
    "equity": 100.0,
    "equity_mark": 100.0,
    "unrealized_pnl": 0.0,
    "peak_equity": 100.0,
    "drawdown_pct": 0.0,
    "total_pnl": 0.0,
    "total_return_pct": 0.0,
    "target_achieved": False,
    "open_positions": [],
    "trade_journal": [],
    "equity_curve": [],
    "scan_results": [],
    "news": {"sentiments": {}, "headlines": [], "all_articles": [],
             "updated_at": None},
    "market_hours": {"india": False, "us": False, "crypto": True,
                     "ist_time": None,
                     "india_holiday": False, "india_holiday_name": None,
                     "us_holiday": False, "us_holiday_name": None},
    "universe": {"total": 0, "counts": {}, "sources": {}, "updated_at": None},
    "agent_kpis": [],
    "rule_stats": {},
    "personas": {"votes_by_symbol": {}, "position_consensus": {},
                 "persona_records": [], "last_updated": None},
    "premarket_briefing": {},
    "memory_log": [],
    "logs": [],
    "last_heartbeat": None,
    "loop_status": {
        "last_outer_run": None,
        "best_metric": None,
        "holdout_sharpe": None,
        "raw_holdout_sharpe": None,
        "last_verdict": None,
        "vetoed": None,
        "downweighted_rules": [],
        "open_hypotheses": [],
        "anchors_frozen": True,
        "metric_definition": ("Out-of-sample walk-forward Sharpe "
                              "(last 20% holdout) w/ hard 2% max-DD veto"),
        "per_module": [],
    },
    "cost_drag_total": 0.0,
    "cost_by_market": {},
    # THE SCOREBOARD — one entry per cohort (isolated paper portfolio).
    "cohorts": [],
    "cohort_count": 0,
    "primary_cohort": "DISTRIBUTED",
    "cycle_count": 0,
    "last_scan": None,
    "errors": [],
}

# WebSocket connections
_connections: list[WebSocket] = []


async def broadcast(data: dict):
    """Push update to all connected dashboard clients."""
    msg = json.dumps(data, default=str)
    dead = []
    for ws in _connections:
        try:
            await ws.send_text(msg)
        except:
            dead.append(ws)
    for ws in dead:
        _connections.remove(ws)


def update_state(key: str, value: Any):
    ENGINE_STATE[key] = value


def append_state(key: str, value: Any):
    if key not in ENGINE_STATE:
        ENGINE_STATE[key] = []
    ENGINE_STATE[key].append(value)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connections.append(ws)
    # Send current state on connect
    await ws.send_text(json.dumps(ENGINE_STATE, default=str))
    try:
        while True:
            await ws.receive_text()  # Keep alive
    except WebSocketDisconnect:
        _connections.remove(ws)


@app.get("/api/state")
async def get_state():
    return ENGINE_STATE


@app.get("/api/trades")
async def get_trades():
    return ENGINE_STATE.get("trade_journal", [])


@app.get("/api/cohorts")
async def get_cohorts():
    """THE SCOREBOARD: every isolated paper portfolio side by side."""
    return {
        "primary": ENGINE_STATE.get("primary_cohort"),
        "count": ENGINE_STATE.get("cohort_count", 0),
        "cohorts": ENGINE_STATE.get("cohorts", []),
    }


@app.get("/api/agents")
async def get_agents():
    return ENGINE_STATE.get("agent_kpis", [])


@app.get("/api/equity")
async def get_equity():
    return ENGINE_STATE.get("equity_curve", [])


@app.get("/api/memory")
async def get_memory():
    return ENGINE_STATE.get("memory_log", [])


@app.get("/api/logs")
async def get_logs(limit: int = 200):
    """The Press Room feed: the shared ring buffer of every log record."""
    return get_recent_logs(max(1, min(limit, 500)))


@app.get("/api/personas")
async def get_personas():
    """The Partners' Room: persona votes for top setups + track records."""
    return ENGINE_STATE.get(
        "personas", {"votes_by_symbol": {}, "position_consensus": {},
                     "persona_records": [], "last_updated": None})


@app.get("/api/rules")
async def get_rules():
    """Every codified investor rule + live weight/track-record from RuleStats."""
    engine = _require_engine()
    book = engine.brain.rulebook
    stats = engine.brain.stats
    rules = []
    for r in book.all():
        s = stats.stats.get(r.id, {})
        wins, losses = s.get("wins", 0), s.get("losses", 0)
        rules.append({
            "id": r.id,
            "investor": r.investor,
            "name": r.name,
            "threshold": r.threshold,
            "category": r.category,
            "sleeve": r.sleeve,
            "confidence": r.confidence,
            "mutability": r.mutability.value,
            "weight": stats.weight(r.id),
            "wins": wins,
            "losses": losses,
            "accuracy": stats.accuracy(r.id),
            "samples": stats.samples(r.id),
        })
    return {
        "total": len(rules),
        "immutable": sum(1 for r in rules if r["mutability"] == "IMMUTABLE"),
        "learnable": sum(1 for r in rules if r["mutability"] == "LEARNABLE"),
        "tracked": sum(1 for r in rules if (r["wins"] + r["losses"]) > 0),
        "rules": rules,
    }


# ── Control endpoints (The Editor's Desk) ──────────────────────


def _require_engine():
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not attached yet")
    return ENGINE


@app.post("/api/config")
async def post_config(cfg: dict[str, Any] = Body(...)):
    """Update engine parameters live (capital, target, interval, limits)."""
    engine = _require_engine()
    applied = engine.apply_config(cfg)
    await engine.push_state()
    return {"ok": True, "applied": applied}


@app.post("/api/scan-now")
async def post_scan_now():
    """Trigger an immediate scan cycle (wakes the run loop). Works while
    paused too: the run loop's `manual` flag lets exactly one cycle through."""
    engine = _require_engine()
    engine.trigger_scan()
    if engine.paused:
        return {"ok": True, "paused": True,
                "message": "Presses paused — scan will run once, then the presses rest"}
    return {"ok": True, "paused": False,
            "message": "Scan triggered — the presses are rolling"}


@app.post("/api/pause")
async def post_pause():
    engine = _require_engine()
    await engine.set_paused(True)
    return {"ok": True, "paused": True}


@app.post("/api/resume")
async def post_resume():
    engine = _require_engine()
    await engine.set_paused(False)
    return {"ok": True, "paused": False}


@app.post("/api/reset-drawdown-baseline")
async def post_reset_drawdown_baseline(body: dict[str, Any] = Body(...)):
    """Re-mark named cohorts' high-water mark to current equity, releasing
    D8's drawdown circuit breaker. Does NOT touch equity/trades/history --
    see AutonomousEngine.reset_drawdown_baseline for the full rationale.
    Requires an explicit cohort list and reason -- no silent defaults."""
    engine = _require_engine()
    cohorts = body.get("cohorts")
    reason = body.get("reason")
    if not cohorts or not isinstance(cohorts, list):
        raise HTTPException(status_code=400, detail="cohorts: list[str] required")
    if not reason:
        raise HTTPException(status_code=400, detail="reason: str required")
    results = await engine.reset_drawdown_baseline(cohorts, reason)
    return {"ok": True, "results": results}


@app.post("/api/fix-position-currency")
async def post_fix_position_currency(body: dict[str, Any] = Body(...)):
    """Correct an open position's mis-tagged currency/fx_rate/market (does
    NOT touch price/size/history). See AutonomousEngine.fix_position_currency."""
    engine = _require_engine()
    cohort = body.get("cohort")
    symbol = body.get("symbol")
    market = body.get("market")
    reason = body.get("reason")
    if not all([cohort, symbol, market, reason]):
        raise HTTPException(status_code=400,
                            detail="cohort, symbol, market, reason all required")
    result = await engine.fix_position_currency(cohort, symbol, market, reason)
    return result


@app.post("/api/universe/push-nse")
async def post_push_nse_universe(body: dict[str, Any] = Body(...)):
    """Local-relay bridge: a machine NOT on NSE's cloud-IP blocklist (see
    data/ingestion/nse_live.py) runs the live-session fetch and pushes
    today's real NSE movers here, since Railway's IP gets 403'd. This is
    TIER 1 of a graceful fallback chain (data/universe.py::_discover_india)
    -- if this never gets called, or the operator's machine is off, the
    engine falls straight back to Railway's own live attempt, then
    bhavcopy, then the hardcoded safety net, exactly as before this existed.

    Requires a shared secret (RELAY_SECRET env var) -- an unauthenticated
    endpoint that injects symbols into a live trading engine's universe
    would let anyone on the internet steer what it trades."""
    import os
    secret = os.environ.get("RELAY_SECRET", "")
    if not secret or body.get("secret") != secret:
        raise HTTPException(status_code=401, detail="invalid or missing secret")
    symbols = body.get("symbols")
    if not symbols or not isinstance(symbols, list):
        raise HTTPException(status_code=400, detail="symbols: list[str] required")
    engine = _require_engine()
    engine.universe.set_india_override(symbols)
    return {"ok": True, "accepted": len(symbols)}


@app.post("/api/restate-trade")
async def post_restate_trade(body: dict[str, Any] = Body(...)):
    """Restate a closed trade's currency-bug-inflated P&L to its true value
    and credit the phantom difference back to equity. Accounting correction,
    not deletion -- see AutonomousEngine.restate_trade."""
    engine = _require_engine()
    cohort = body.get("cohort")
    trade_id = body.get("trade_id")
    fx_rate = body.get("fx_rate")
    reason = body.get("reason")
    if not all([cohort, trade_id, fx_rate, reason]):
        raise HTTPException(status_code=400,
                            detail="cohort, trade_id, fx_rate, reason all required")
    return await engine.restate_trade(cohort, trade_id, float(fx_rate), reason)


@app.post("/api/admin/reset")
async def post_admin_reset(body: dict[str, Any] = Body(...)):
    """CLEAN-SLATE reset of the whole book. DESTRUCTIVE — wipes trades,
    positions, equity curve, memory log, audit events and attribution (DB +
    in-memory) and re-seeds every cohort at its starting capital.

    Guarded by the SAME shared secret as /api/universe/push-nse (RELAY_SECRET
    env var): the body must carry a matching `secret`, otherwise an anonymous
    caller could zero out a live engine's entire record.

    Body:
      secret         (str, required)  — must equal RELAY_SECRET.
      reason         (str, required)  — recorded in the RESET audit event.
      reset_learning (bool, default False) — if True, ALSO wipe the learned
                     rule weights + attribution so the brain starts fresh;
                     if False, the learned weights are KEPT.
    """
    import os
    secret = os.environ.get("RELAY_SECRET", "")
    if not secret or body.get("secret") != secret:
        raise HTTPException(status_code=401, detail="invalid or missing secret")
    reason = body.get("reason")
    if not reason:
        raise HTTPException(status_code=400, detail="reason: str required")
    reset_learning = bool(body.get("reset_learning", False))
    engine = _require_engine()
    return await engine.reset_all(reset_learning=reset_learning, reason=reason)


@app.post("/api/refresh-news")
async def post_refresh_news():
    """Force a fresh news/macro sweep, bypassing the cache."""
    engine = _require_engine()
    await engine.refresh_macro_intel(force=True)
    await engine.push_state()
    return {"ok": True, "news": ENGINE_STATE.get("news")}


@app.get("/api/report", response_class=HTMLResponse)
async def get_report(days: int = 7, cohort: str = "all"):
    """Equity-research-grade weekly audit note (print-ready HTML → PDF)."""
    from reports.research_note import build_report
    days = max(1, min(days, 90))
    try:
        return build_report(ENGINE_STATE, days=days, cohort=cohort)
    except Exception as e:
        return HTMLResponse(f"<h1>Report error</h1><pre>{e}</pre>", status_code=500)


@app.get("/api/report.csv")
async def get_report_csv(days: int = 7, cohort: str = "all"):
    """Closed-trade blotter as CSV."""
    from fastapi.responses import PlainTextResponse
    from reports.research_note import build_csv
    days = max(1, min(days, 90))
    csv_text = build_csv(days=days, cohort=cohort)
    return PlainTextResponse(csv_text, headers={
        "Content-Disposition": f'attachment; filename="blotter_{days}d.csv"',
        "Content-Type": "text/csv",
    })


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


# ── BACKTEST AUDIT: offline naive-vs-learned validation ──────────────────
# Reuses the SAME production score_asset/RuleBrain/PersonaEngine/cost-gate
# pipeline against real historical data (backtest/learning_validation.py).
# Runs as a background task, offloaded to a thread for the CPU-bound
# simulation, so it never blocks the live Scout/Sentry loops. Guarded by a
# simple in-progress flag against concurrent runs.

_BACKTEST_STATE: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "last_report": None,       # full structured report dict (in-memory)
    "last_json_path": None,
    "last_md_path": None,
}


async def _run_backtest_background(lookback_days: int):
    from backtest.learning_validation import run_learning_validation, write_report

    _BACKTEST_STATE["running"] = True
    _BACKTEST_STATE["error"] = None
    _BACKTEST_STATE["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        report = await run_learning_validation(lookback_days)
        json_path, md_path = write_report(report)
        _BACKTEST_STATE["last_report"] = report
        _BACKTEST_STATE["last_json_path"] = str(json_path)
        _BACKTEST_STATE["last_md_path"] = str(md_path)
    except Exception as e:
        _BACKTEST_STATE["error"] = str(e)
    finally:
        _BACKTEST_STATE["running"] = False
        _BACKTEST_STATE["finished_at"] = datetime.now(timezone.utc).isoformat()


def _backtest_summary() -> dict:
    rep = _BACKTEST_STATE.get("last_report")
    summary = None
    if rep:
        summary = {
            "verdict": rep.get("verdict"),
            "comparison": rep.get("comparison"),
            "scope": {k: v for k, v in rep.get("scope", {}).items() if k != "fetch_reports"},
            "live_rule_stats_untouched": rep.get("live_rule_stats_untouched"),
        }
    return {
        "running": _BACKTEST_STATE["running"],
        "started_at": _BACKTEST_STATE["started_at"],
        "finished_at": _BACKTEST_STATE["finished_at"],
        "error": _BACKTEST_STATE["error"],
        "last_json_path": _BACKTEST_STATE["last_json_path"],
        "last_md_path": _BACKTEST_STATE["last_md_path"],
        "summary": summary,
    }


@app.get("/api/backtest/learning-validation")
async def get_backtest_learning_validation(lookback_days: int = 730):
    """Trigger the naive-vs-learned validation run in the background (takes
    several minutes — data pull + two full walk-forward simulations) and
    return the last COMPLETED report's summary + where to find the full
    file. Poll this same endpoint (or /api/backtest/status) for progress."""
    if _BACKTEST_STATE["running"]:
        return {"ok": True, "triggered": False, "reason": "already running",
                **_backtest_summary()}
    lookback_days = max(60, min(lookback_days, 3650))
    asyncio.create_task(_run_backtest_background(lookback_days))
    return {"ok": True, "triggered": True, **_backtest_summary()}


@app.get("/api/backtest/status")
async def get_backtest_status():
    """Lightweight polling endpoint for the dashboard's run/idle indicator."""
    return _backtest_summary()


@app.get("/api/backtest/report")
async def get_backtest_report():
    """Full structured trade-by-trade JSON (naive + learned) for the
    BACKTEST AUDIT dashboard tab to render/filter client-side. Falls back to
    the last report written to disk if the in-memory copy was lost (process
    restart) but a completed run exists under reports/backtest/."""
    rep = _BACKTEST_STATE.get("last_report")
    if rep is None:
        latest = Path("reports/backtest/latest.json")
        if latest.exists():
            try:
                rep = json.loads(latest.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                rep = None
    if rep is None:
        return {"ok": False, "status": _backtest_summary(), "report": None}
    return {"ok": True, "status": _backtest_summary(), "report": rep}
