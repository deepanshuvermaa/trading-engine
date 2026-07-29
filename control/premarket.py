"""Pre-market briefing — "read the report, discuss, watch the open, decide".

How real discretionary desks actually start the day, made deterministic:

1. READ YESTERDAY'S REPORT — reuse reports/research_note.py's audit-trail
   collector to summarize what closed, what was learned.
2. DISCUSS CANDIDATE NAMES — run today's live universe discovery, then the
   FULL existing persona vote (personas/engine.py PersonaEngine, via
   AutonomousEngine.score_asset — same code path real trading uses) on
   whatever data is available pre-open (yesterday's daily bar; an early
   Finnhub quote for US names when a key is configured).
3. RANK -> WATCHLIST — keep the top-N by persona consensus strength. This
   does NOT open any positions; it only seeds a watchlist + an opening-
   candle confirmation gate that autonomous.py's Scout loop consults.

No LLM anywhere in this path — every step reuses the engine's existing
deterministic scoring/persona machinery.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from utils.logger import get_logger

log = get_logger("control.premarket")

IST = ZoneInfo("Asia/Kolkata")
US_EASTERN = ZoneInfo("America/New_York")

DEFAULT_TOP_N = 10
CANDIDATE_CAP = 30          # bound premarket scoring runtime per market
LOOKBACK_DAYS = 260         # enough for SMA200 + 52w hi/lo
# Watchlist admission bar, deliberately looser than live entry's 25 -- this
# only decides "worth the panel's attention today," never opens a position.
WATCHLIST_MIN_SCORE = 15.0


class PremarketBriefing:
    """Runs once per market per trading day, at/after that market's open.
    Populates engine.premarket_gate (opening-candle confirmation state,
    consulted by the Scout loop) and engine.premarket_state (the briefing
    itself, surfaced on ENGINE_STATE["premarket_briefing"])."""

    def __init__(self, engine):
        self.engine = engine
        self._last_run_date: dict[str, str] = {}   # market -> "YYYY-MM-DD" (local)

    # ── scheduling ───────────────────────────────────────────────

    @staticmethod
    def _local_date(market: str, now: datetime) -> str:
        tz = IST if market == "india" else US_EASTERN
        return now.astimezone(tz).strftime("%Y-%m-%d")

    def _due(self, market: str, hours: dict, now: datetime) -> bool:
        """Due once the market is open and we haven't briefed it yet today
        (local trading-day dedup, so a restart never double-fires)."""
        if not hours.get(market):
            return False
        today = self._local_date(market, now)
        return self._last_run_date.get(market) != today

    async def maybe_run(self, top_n: int = DEFAULT_TOP_N) -> dict | None:
        """Called once per Scout cycle. Non-blocking best-effort: any error
        here must never take down the trading loop."""
        from autonomous import market_hours_status  # local import: avoid cycle

        now = datetime.now(timezone.utc)
        hours = market_hours_status(now)
        ran: dict | None = None
        for market in ("india", "us"):
            if self._due(market, hours, now):
                try:
                    ran = await self.run(market, top_n=top_n)
                except Exception as e:
                    log.warning(f"Premarket briefing [{market}] failed: {e}")
                    # Still mark as attempted today so a persistent failure
                    # doesn't retry every single cycle.
                    self._last_run_date[market] = self._local_date(market, now)
        return ran

    # ── the briefing itself ─────────────────────────────────────

    def _yesterday_report(self, days: int = 2) -> dict:
        """Step 1: read yesterday's closed-trade summary + lessons, reusing
        research_note.py's own audit-trail collector (no reinvention)."""
        try:
            from reports.research_note import _collect, _metrics
            d = _collect(days)
            m = _metrics(d["attrib"])
            lessons = (d.get("lessons") or "").strip().splitlines()
            return {
                "trades_closed": m["n"],
                "wins": m["wins"],
                "losses": m["losses"],
                "win_rate": round(m["win_rate"], 1),
                "net_pnl": round(m["net_pnl"], 4),
                "profit_factor": round(m["profit_factor"], 2) if m["profit_factor"] not in (None, float("inf")) else None,
                "recent_lessons": lessons[-5:],
            }
        except Exception as e:
            log.warning(f"Premarket: could not read yesterday's report ({e})")
            return {"trades_closed": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                    "net_pnl": 0.0, "profit_factor": None, "recent_lessons": []}

    async def _score_candidate(self, market: str, symbol: str) -> dict | None:
        """Step 3: full persona vote on whatever data is available pre-open —
        yesterday's daily OHLCV (via the EXISTING per-market provider), plus
        an early Finnhub quote for US names when a key is configured (used
        only to compute a gap%; scoring itself still runs on the daily bar,
        same code path as real trading via AutonomousEngine.score_asset)."""
        engine = self.engine
        provider = {"india": engine.indian_eq, "us": engine.us_eq}.get(market)
        if provider is None:
            return None

        start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
        try:
            df = await provider.get_ohlcv(symbol, "1d", start, use_cache=True)
        except Exception as e:
            log.debug(f"Premarket: OHLCV fetch failed for {symbol}: {e}")
            return None
        if df is None or len(df) < 50:
            return None

        # Reuse the EXISTING score_asset -> RuleBrain -> PersonaEngine path,
        # scored against the primary cohort's book (read-only: this never
        # opens a position, only produces the same setup dict real scans do).
        # A looser bar than live entry (25): this only builds a WATCHLIST, it
        # never opens a position. The opening-candle confirmation gate (see
        # run_cycle) is the real check before any actual entry. Going lower
        # than this would start admitting pure noise the persona panel can't
        # meaningfully discuss; going all the way down to 0 would defeat the
        # point of having a bar at all -- 15 is the inversion-tested middle:
        # loose enough to surface real but developing setups, not so loose
        # the "morning discussion" is just reading random tickers.
        try:
            result = engine.score_asset(
                engine.primary, symbol, df, macro=engine.macro_snapshot,
                min_score=WATCHLIST_MIN_SCORE)
        except Exception as e:
            log.debug(f"Premarket: score_asset failed for {symbol}: {e}")
            return None
        if not result:
            return None

        gap_pct = None
        if market == "us" and getattr(engine, "finnhub", None) is not None:
            try:
                live_px = await engine.finnhub.get_price(symbol, "us")
                prior_close = result.get("current_price")
                if live_px and prior_close:
                    gap_pct = round((live_px - prior_close) / prior_close * 100, 2)
            except Exception:
                gap_pct = None

        result["gap_pct"] = gap_pct
        return result

    async def run(self, market: str, top_n: int = DEFAULT_TOP_N) -> dict:
        """Build and store today's briefing for one market. Never raises —
        any stage failure degrades that section to empty and is logged."""
        engine = self.engine
        now = datetime.now(timezone.utc)
        today_local = self._local_date(market, now)
        self._last_run_date[market] = today_local

        # 1. Read yesterday's report.
        yesterday = self._yesterday_report()

        # 2. Today's live candidate list (Part 2's live-NSE-first discovery
        # for india; yfinance screeners for us — data/universe.py already
        # does this, we just read its cached/fresh result).
        try:
            uni = await engine.universe.discover()
        except Exception as e:
            log.warning(f"Premarket [{market}]: universe discovery failed: {e}")
            uni = {}
        candidates = list(uni.get(market, []))[:CANDIDATE_CAP]

        # 3. The agents discuss — full persona vote per candidate.
        scored: list[dict] = []
        votes_by_symbol: dict[str, Any] = {}
        for sym in candidates:
            result = await self._score_candidate(market, sym)
            if not result:
                continue
            consensus = result.get("persona_consensus") or {}
            action = consensus.get("action", "HOLD")
            if action not in ("BUY", "SELL"):
                continue  # no working conviction — not watchlist material
            scored.append({
                "symbol": sym,
                "market": market,
                "direction": action,
                "strength": consensus.get("strength", 0),
                "consensus": consensus.get("consensus", 0.0),
                "score": result.get("score"),
                "current_price": result.get("current_price"),
                "gap_pct": result.get("gap_pct"),
                "summary": consensus.get("summary", ""),
                "top_reasons": (result.get("reasons") or [])[:3],
            })
            votes_by_symbol[sym] = {
                "direction": action,
                "votes": result.get("persona_votes", []),
                "risk": result.get("persona_risk"),
                "consensus": consensus,
            }

        # 4. Rank by consensus strength, keep the top N as today's watchlist.
        scored.sort(key=lambda r: r["strength"], reverse=True)
        watchlist = scored[:max(1, top_n)]

        # Seed the opening-candle confirmation gate (consulted by the Scout
        # loop in autonomous.py's run_cycle) — PENDING until the first
        # completed post-open candle is observed.
        for w in watchlist:
            engine.premarket_gate[w["symbol"]] = {
                "date": today_local,
                "market": market,
                "direction": w["direction"],
                "status": "PENDING",
                "transition_cycle": None,
                "opening_candle": None,
            }

        record = {
            "market": market,
            "watchlist": watchlist,
            "votes": votes_by_symbol,
            "confirmed": [],
            "rejected": [],
            "yesterday": yesterday,
            "candidates_scanned": len(candidates),
            "generated_at": now.isoformat(),
        }
        engine.premarket_state[market] = record
        engine._mem(
            f"PREMARKET BRIEFING [{market}]: {len(watchlist)} name(s) on today's "
            f"watchlist out of {len(candidates)} candidates scanned "
            f"(yesterday: {yesterday['trades_closed']} trades, "
            f"{yesterday['win_rate']:.0f}% win rate, "
            f"net ${yesterday['net_pnl']:+.2f})",
            "SUCCESS" if watchlist else "info")
        return record
