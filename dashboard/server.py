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
                     "ist_time": None},
    "universe": {"total": 0, "counts": {}, "sources": {}, "updated_at": None},
    "agent_kpis": [],
    "rule_stats": {},
    "personas": {"votes_by_symbol": {}, "position_consensus": {},
                 "persona_records": [], "last_updated": None},
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


@app.post("/api/refresh-news")
async def post_refresh_news():
    """Force a fresh news/macro sweep, bypassing the cache."""
    engine = _require_engine()
    await engine.refresh_macro_intel(force=True)
    await engine.push_state()
    return {"ok": True, "news": ENGINE_STATE.get("news")}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")
