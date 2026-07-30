"""Local NSE live-data relay.

Railway's server IP gets 403'd by NSE's anti-bot filtering (confirmed --
NSE blocks known cloud-provider ranges). This same live-session fetch works
fine from a residential/local machine (verified earlier this session: 60
real live symbols pulled). This script runs LOCALLY, fetches NSE's actual
live gainers/losers/most-active, and pushes the symbol list to the live
Railway engine so the trading system sees TODAY's real movers instead of
yesterday's bhavcopy.

This is TIER 1 of a fallback chain (data/universe.py::_discover_india) --
if this script isn't running (laptop off, no network), the engine falls
back to its own live attempt, then bhavcopy, then a hardcoded safety net.
Nothing breaks if you don't run this; it only helps when you do.

Usage:
    set RAILWAY_URL=https://dalal-street.up.railway.app
    set RELAY_SECRET=<same value as the RELAY_SECRET env var on Railway>
    python scripts/nse_local_relay.py

Runs continuously: sleeps until the next NSE session (09:15-15:30 IST,
Mon-Fri), then pushes fresh data every ~5 minutes during market hours.
Ctrl+C to stop -- the live engine just falls back to bhavcopy from then on.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from data.ingestion.nse_live import NSELiveSession
from utils.logger import get_logger

log = get_logger("nse_relay")

IST = ZoneInfo("Asia/Kolkata")
PUSH_INTERVAL_SECONDS = 300  # every 5 min during market hours
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)


def _market_open_now(now_ist: datetime) -> bool:
    if now_ist.weekday() >= 5:
        return False
    hm = (now_ist.hour, now_ist.minute)
    return MARKET_OPEN <= hm < MARKET_CLOSE


def _seconds_to_next_open(now_ist: datetime) -> float:
    """Seconds until the next 09:15 IST on a weekday."""
    target = now_ist.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1],
                              second=0, microsecond=0)
    if now_ist >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return max(1.0, (target - now_ist).total_seconds())


async def fetch_and_push(base_url: str, secret: str) -> int:
    """One fetch+push cycle. Returns symbol count pushed, 0 on any failure
    (never raises -- a bad cycle just means we retry next interval)."""
    session = NSELiveSession()
    try:
        gainers_losers = await session.get_live_gainers_losers()
        most_active = await session.get_live_most_active()
    except Exception as e:
        log.warning(f"NSE live fetch failed this cycle: {e}")
        return 0
    finally:
        await session.close()

    symbols: list[str] = []
    for bucket in ("gainers", "losers"):
        for row in gainers_losers.get(bucket, [])[:15]:
            sym = row.get("symbol") or row.get("SYMBOL")
            if sym:
                symbols.append(sym)
    for row in most_active[:15]:
        sym = row.get("symbol") or row.get("SYMBOL")
        if sym:
            symbols.append(sym)

    symbols = list(dict.fromkeys(symbols))  # dedupe, keep order
    if not symbols:
        log.warning("NSE live fetch returned 0 symbols this cycle")
        return 0

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/universe/push-nse",
                json={"symbols": symbols, "secret": secret},
            )
            resp.raise_for_status()
        log.info(f"Pushed {len(symbols)} live NSE symbols to {base_url}")
        return len(symbols)
    except Exception as e:
        log.error(f"Push to Railway failed: {e}")
        return 0


async def main():
    base_url = os.environ.get("RAILWAY_URL", "").strip()
    secret = os.environ.get("RELAY_SECRET", "").strip()
    if not base_url or not secret:
        log.error("Set RAILWAY_URL and RELAY_SECRET env vars first. See module docstring.")
        return

    log.info(f"NSE local relay starting -> {base_url}")
    while True:
        now_ist = datetime.now(IST)
        if not _market_open_now(now_ist):
            wait_s = _seconds_to_next_open(now_ist)
            log.info(f"NSE closed. Sleeping {wait_s/3600:.1f}h until next session.")
            await asyncio.sleep(min(wait_s, 3600))  # wake hourly to re-check, avoid oversleeping
            continue

        await fetch_and_push(base_url, secret)
        await asyncio.sleep(PUSH_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Relay stopped. Live engine will fall back to bhavcopy from here.")
