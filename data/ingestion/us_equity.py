"""US equity data via yfinance."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
import yfinance as yf

from data.models import Market
from data.ingestion.base import DataProvider
from utils.logger import get_logger

log = get_logger("data.us_equity")

# Map our timeframes to yfinance intervals
YF_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "1h",  # yfinance doesn't have 4h — we resample
    "1d": "1d", "1w": "1wk",
}


class USEquityProvider(DataProvider):
    market = Market.US_EQUITY

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_sync, symbol, timeframe, start, end
        )

    async def fetch_intraday(
        self, symbol: str, interval: str = "15m", lookback_days: int = 30
    ) -> pd.DataFrame:
        """Intraday bars via yfinance (period-based; last ~60 days for 15m)."""
        from data.ingestion.intraday import fetch_yf_intraday

        return await fetch_yf_intraday(symbol, interval, lookback_days)

    def _fetch_sync(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None,
    ) -> pd.DataFrame:
        interval = YF_INTERVAL.get(timeframe, "1d")
        needs_resample = (timeframe == "4h")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d") if end else None,
                interval="1h" if needs_resample else interval,
                auto_adjust=True,
            )
        except Exception as e:
            log.error(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "timestamp"

        if needs_resample:
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        log.info(f"Fetched {len(df)} bars for {symbol} {timeframe} (yfinance)")
        return df
