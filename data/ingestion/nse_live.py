"""NSE live-session client — today's REAL intraday movers, not yesterday's
bhavcopy.

NSE's public website serves live JSON off internal endpoints, but only to
clients that first "browse" nseindia.com like a real browser to pick up
session cookies, then replay those cookies + browser-like headers on the
data calls. This is the same technique documented by the
hi-imcodeman/stock-nse-india project. NSE blocks aggressively (rotates
cookies, 401/403s bots, occasionally just hangs) so every call here is
best-effort, timeout-bounded, and NEVER raises into the caller — an empty
result just means "today's live path didn't work this cycle", and the
caller (data/universe.py) falls back to the bhavcopy path, which remains
the safety net.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from utils.logger import get_logger

log = get_logger("data.nse_live")

BASE = "https://www.nseindia.com"
GAINERS_LOSERS_URL = f"{BASE}/api/live-analysis-variations"
MOST_ACTIVE_URL = f"{BASE}/api/live-analysis-most-active-securities"
INDEX_CONSTITUENTS_URL = f"{BASE}/api/equity-stockIndices"

REQUEST_TIMEOUT = 12.0
COOKIE_TTL = 240.0  # seconds — refresh session periodically even without a 401/403

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


class NSELiveSession:
    """Acquires + refreshes NSE session cookies, then hits the live-analysis
    JSON endpoints. Best-effort only: NSE blocks datacenter IPs (Railway,
    most cloud hosts) aggressively, so a clean empty result here is expected
    and handled — the caller's bhavcopy fallback carries the day."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._cookies_at: float = 0.0

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_session(self, force: bool = False) -> bool:
        """GET the NSE homepage (and a warm-up data page) to acquire cookies.
        Returns True if we now have cookies to work with."""
        client = self._get_client()
        now = time.monotonic()
        if (not force and client.cookies
                and (now - self._cookies_at) < COOKIE_TTL):
            return True
        try:
            resp = await client.get(BASE, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                log.warning(f"NSE live: homepage GET returned {resp.status_code}")
                return False
            # A second hit on a data-bearing page tends to set the remaining
            # cookies the JSON endpoints check for.
            await client.get(f"{BASE}/market-data/live-equity-market",
                             timeout=REQUEST_TIMEOUT)
            self._cookies_at = time.monotonic()
            return bool(client.cookies)
        except httpx.HTTPError as e:
            log.warning(f"NSE live: session acquisition failed: {e}")
            return False
        except Exception as e:
            log.warning(f"NSE live: unexpected session error: {e}")
            return False

    async def _get_json(self, url: str, params: dict | None = None,
                        _retried: bool = False) -> dict | None:
        client = self._get_client()
        if not await self._ensure_session():
            return None
        try:
            resp = await client.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (401, 403):
                if not _retried:
                    log.info(f"NSE live: {resp.status_code} on {url} — "
                             f"refreshing session and retrying once")
                    if await self._ensure_session(force=True):
                        return await self._get_json(url, params, _retried=True)
                log.warning(f"NSE live: {resp.status_code} on {url} after retry — giving up")
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            log.warning(f"NSE live: request error on {url}: {e}")
            return None
        except ValueError as e:  # JSON decode
            log.warning(f"NSE live: bad JSON from {url}: {e}")
            return None
        except Exception as e:
            log.warning(f"NSE live: unexpected error on {url}: {e}")
            return None

    async def get_live_gainers_losers(self) -> dict[str, list[dict[str, Any]]]:
        """Today's live top gainers + losers (NIFTY-wide) from
        live-analysis-variations. Returns {"gainers": [...], "losers": [...]}
        with each item {"symbol", "ltp", "pct_change"} — empty on failure."""
        out: dict[str, list[dict[str, Any]]] = {"gainers": [], "losers": []}
        # NSE's own endpoint spells the losers param "loosers" (double-o) —
        # verified against the live API; "losers" 404s with "Missing index
        # or key". Kept as a literal mapping so this stays obvious/auditable.
        nse_index_param = {"gainers": "gainers", "losers": "loosers"}
        for side in ("gainers", "losers"):
            data = await self._get_json(
                GAINERS_LOSERS_URL, params={"index": nse_index_param[side]})
            if not data:
                continue
            rows = (data.get("NIFTY", {}) or {}).get("data") or data.get("data") or []
            if not isinstance(rows, list):
                log.warning(f"NSE live: unexpected '{side}' payload shape ({type(rows).__name__}) — skipping")
                continue
            picks = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                sym = (r.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                try:
                    pct = float(r.get("perChange") or r.get("pChange") or 0)
                    ltp = float(r.get("ltp") or r.get("lastPrice") or 0)
                except (TypeError, ValueError):
                    pct, ltp = 0.0, 0.0
                picks.append({"symbol": sym, "ltp": ltp, "pct_change": pct})
            out[side] = picks
        return out

    async def get_live_most_active(self) -> list[dict[str, Any]]:
        """Most-active-by-value securities right now, if NSE exposes the
        endpoint reliably. Returns [] on any failure."""
        data = await self._get_json(MOST_ACTIVE_URL, params={"index": "value"})
        if not data:
            return []
        rows = data.get("data") or []
        if not isinstance(rows, list):
            return []
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            sym = (r.get("symbol") or "").strip().upper()
            if not sym:
                continue
            try:
                val = float(r.get("totalTradedValue") or r.get("value") or 0)
                pct = float(r.get("perChange") or r.get("pChange") or 0)
            except (TypeError, ValueError):
                val, pct = 0.0, 0.0
            out.append({"symbol": sym, "value": val, "pct_change": pct})
        return out

    async def get_index_variations(self, index: str = "NIFTY 500") -> list[dict[str, Any]]:
        """Live %-change for every constituent of an index (e.g. NIFTY 500) —
        the broadest live-mover source when it responds. Empty on failure."""
        data = await self._get_json(INDEX_CONSTITUENTS_URL, params={"index": index})
        if not data:
            return []
        rows = data.get("data") or []
        if not isinstance(rows, list):
            return []
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            sym = (r.get("symbol") or "").strip().upper()
            if not sym or sym == index.upper().replace(" ", ""):
                continue
            try:
                pct = float(r.get("pChange") or 0)
                ltp = float(r.get("lastPrice") or 0)
            except (TypeError, ValueError):
                pct, ltp = 0.0, 0.0
            out.append({"symbol": sym, "ltp": ltp, "pct_change": pct})
        return out


# Module-level singleton (one cookie jar, reused across universe-discovery
# cycles) — mirrors the pattern in data/universe.py / finnhub_client.py.
_session: NSELiveSession | None = None


def get_nse_live_session() -> NSELiveSession:
    global _session
    if _session is None:
        _session = NSELiveSession()
    return _session
