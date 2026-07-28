"""Finnhub live-quote client — US equities + crypto mark-to-market only.

Reads FINNHUB_API_KEY from the environment (never hardcoded, never
committed). If the key is unset, every call reports itself "unavailable"
and returns None; nothing raises into the caller. Free-tier rate limits
(~60 calls/min) are respected with a simple token-bucket, and each quote is
cached for CACHE_TTL seconds so the fast Sentry loop doesn't hammer the API.

NSE (Indian equities) is deliberately NOT supported here — Finnhub's free
tier has no access to NSE symbols (confirmed: returns "You don't have
access to this resource"). Callers should keep using the existing yfinance
path for India.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from utils.logger import get_logger

log = get_logger("data.finnhub")

BASE_URL = "https://finnhub.io/api/v1/quote"
CACHE_TTL = 25.0          # seconds — well inside free-tier rate limits
RATE_LIMIT_PER_MIN = 55   # stay under Finnhub's free-tier ~60/min ceiling
REQUEST_TIMEOUT = 8.0


def _binance_symbol(pair: str) -> str:
    """Map our SYMBOL/USDT style pair to Finnhub's Binance-prefixed form.
    BTC/USDT -> BINANCE:BTCUSDT. Any other quote currency is best-effort
    normalized to USDT (Finnhub's Binance feed is USDT-quoted)."""
    base, _, quote = pair.partition("/")
    quote = (quote or "USDT").upper().replace("USD", "USDT")
    if not quote.endswith("USDT"):
        quote = "USDT"
    return f"BINANCE:{base.upper()}{quote}"


class _TokenBucket:
    """Gentle sleep-based throttle — never more than `rate` calls/min."""

    def __init__(self, rate_per_min: int):
        self.min_interval = 60.0 / max(1, rate_per_min)
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


class FinnhubClient:
    """Async Finnhub quote client. Never raises — callers get None on any
    failure (no key, rate-limited, network error, bad symbol)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key if api_key is not None
                         else os.environ.get("FINNHUB_API_KEY", "")).strip()
        self._bucket = _TokenBucket(RATE_LIMIT_PER_MIN)
        self._cache: dict[str, tuple[float, dict]] = {}
        self._client: httpx.AsyncClient | None = None
        self._warned_no_key = False

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def to_finnhub_symbol(engine_symbol: str, market: str) -> str | None:
        """Map an engine symbol to Finnhub's expected format. Returns None
        for markets Finnhub free-tier can't serve (India/NSE)."""
        if market == "crypto":
            return _binance_symbol(engine_symbol)
        if market == "us":
            return engine_symbol.upper()
        return None  # india: no free-tier access — caller must fall back

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """GET /quote for a Finnhub-formatted symbol (already mapped via
        to_finnhub_symbol). Returns {"c","h","l","o","pc","t"} or None.
        Cached CACHE_TTL seconds; never raises."""
        if not self.available:
            if not self._warned_no_key:
                log.info("Finnhub: FINNHUB_API_KEY not set — live quotes "
                         "unavailable, falling back to existing data path")
                self._warned_no_key = True
            return None

        now = time.monotonic()
        hit = self._cache.get(symbol)
        if hit is not None and (now - hit[0]) < CACHE_TTL:
            return hit[1]

        try:
            await self._bucket.wait()
            client = self._get_client()
            resp = await client.get(
                BASE_URL, params={"symbol": symbol, "token": self.api_key})
            if resp.status_code == 429:
                log.warning(f"Finnhub: rate-limited on {symbol} — "
                           f"falling back this cycle")
                return None
            if resp.status_code == 403:
                log.warning(f"Finnhub: 403 on {symbol} (no access on this "
                           f"tier?) — falling back")
                return None
            resp.raise_for_status()
            data = resp.json()
            # Finnhub returns all-zero payload for unknown/inaccessible symbols.
            if not data or data.get("c") in (None, 0):
                log.debug(f"Finnhub: empty/zero quote for {symbol} — "
                         f"falling back")
                return None
            self._cache[symbol] = (now, data)
            return data
        except httpx.HTTPError as e:
            log.warning(f"Finnhub: request error for {symbol}: {e} — "
                       f"falling back")
            return None
        except Exception as e:  # never raise into the caller
            log.warning(f"Finnhub: unexpected error for {symbol}: {e} — "
                       f"falling back")
            return None

    async def get_price(self, engine_symbol: str, market: str) -> float | None:
        """Convenience: engine symbol + market -> live last price, or None
        if unavailable/unsupported (india) — caller falls back to yfinance."""
        fh_symbol = self.to_finnhub_symbol(engine_symbol, market)
        if fh_symbol is None:
            return None
        quote = await self.get_quote(fh_symbol)
        if not quote:
            return None
        px = quote.get("c")
        try:
            px = float(px)
        except (TypeError, ValueError):
            return None
        return px if px > 0 else None


# Module-level singleton — one client, one cache, one rate limiter, shared
# by whatever imports this module (mirrors the pattern in data/universe.py).
_client: FinnhubClient | None = None


def get_finnhub_client() -> FinnhubClient:
    global _client
    if _client is None:
        _client = FinnhubClient()
    return _client
