"""Shared intraday (15m) OHLCV fetcher via yfinance.

All three markets fetch intraday bars from yfinance:
  - crypto    BTC/USDT -> BTC-USD
  - us        AAPL     -> AAPL
  - india     RELIANCE -> RELIANCE.NS

yfinance serves 15m bars for roughly the last 60 calendar days and is delayed
~15 minutes and rate-limits aggressively. This module therefore:
  - caches each (symbol, interval, lookback) result for `CACHE_TTL` seconds so the
    45s Sentry loop and the candle-aligned Scout loop never hammer the wire;
  - retries with gentle backoff on transient failures;
  - returns an EMPTY DataFrame (never raises) so a single bad symbol is skipped,
    not fatal.
"""

from __future__ import annotations

import asyncio
import time

import pandas as pd

from utils.logger import get_logger

log = get_logger("data.intraday")

CACHE_TTL = 60.0          # seconds — Sentry re-uses bars within a minute
YF_INTRADAY_MAX_DAYS = 59  # yfinance caps 15m history at ~60 calendar days

# key "symbol|interval|lookback" -> (monotonic_ts, DataFrame)
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def _fetch_sync(yf_symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """Blocking yfinance intraday fetch with retry/backoff. Never raises."""
    import yfinance as yf

    days = min(max(int(lookback_days), 1), YF_INTRADAY_MAX_DAYS)
    period = f"{days}d"
    last_err: Exception | None = None

    for attempt in range(3):
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]]
                df.index.name = "timestamp"
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                if len(df):
                    return df
            # Empty result — treat as a soft miss, retry once or twice
            last_err = None
        except Exception as e:  # network / rate-limit / parse
            last_err = e
        time.sleep(1.2 * (attempt + 1))  # gentle backoff (thread, so blocking is fine)

    if last_err is not None:
        log.warning(f"yfinance intraday failed for {yf_symbol} {interval}: {last_err}")
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


async def fetch_yf_intraday(
    yf_symbol: str,
    interval: str = "15m",
    lookback_days: int = 30,
    cache_ttl: float = CACHE_TTL,
) -> pd.DataFrame:
    """Return cached-or-fresh intraday OHLCV for a yfinance ticker.

    Returns an empty DataFrame on failure (caller should skip the asset)."""
    key = f"{yf_symbol}|{interval}|{lookback_days}"
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and (now - hit[0]) < cache_ttl:
        return hit[1].copy()

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None, _fetch_sync, yf_symbol, interval, lookback_days)
    if df is not None and len(df):
        _CACHE[key] = (now, df.copy())
    return df
