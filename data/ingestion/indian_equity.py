"""Indian equity data via jugaad-data (NSE/BSE official archives)."""

from __future__ import annotations

import asyncio
from datetime import datetime, date

import pandas as pd

from data.models import Market
from data.ingestion.base import DataProvider
from utils.logger import get_logger

log = get_logger("data.indian_eq")


class IndianEquityProvider(DataProvider):
    market = Market.INDIAN_EQUITY

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        # jugaad-data is synchronous — run in executor
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, self._fetch_sync, symbol, start, end
        )
        return df

    async def fetch_intraday(
        self, symbol: str, interval: str = "15m", lookback_days: int = 30
    ) -> pd.DataFrame:
        """Intraday bars via yfinance .NS symbols (jugaad-data is daily-only).
        RELIANCE -> RELIANCE.NS."""
        from data.ingestion.intraday import fetch_yf_intraday

        yf_symbol = symbol if symbol.upper().endswith(".NS") else f"{symbol}.NS"
        return await fetch_yf_intraday(yf_symbol, interval, lookback_days)

    def _fetch_sync(
        self, symbol: str, start: datetime, end: datetime | None
    ) -> pd.DataFrame:
        from jugaad_data.nse import stock_df

        start_date = start.date() if isinstance(start, datetime) else start
        end_date = (end.date() if end else date.today())

        try:
            df = stock_df(
                symbol=symbol,
                from_date=start_date,
                to_date=end_date,
                series="EQ",
            )
        except Exception as e:
            log.error(f"jugaad-data error for {symbol}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Drop columns that are ALREADY duplicated under their raw jugaad-data
        # names (rare, but happens).
        df = df.loc[:, ~df.columns.duplicated()]

        # jugaad-data column names vary — normalize. NOTE: multiple distinct
        # raw columns can map to the SAME target (e.g. "CLOSE" and "LTP" both
        # -> "close") -- that duplication doesn't exist until AFTER this
        # rename, so the dedup above (which runs on the pre-rename names)
        # cannot catch it. A second dedup below, post-rename, is required --
        # its absence was crashing every NSE daily-bar fetch with "Duplicate
        # column names found" the moment jugaad-data returned both columns.
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("open", "open price"):
                col_map[col] = "open"
            elif cl in ("high", "high price"):
                col_map[col] = "high"
            elif cl in ("low", "low price"):
                col_map[col] = "low"
            elif cl in ("close", "close price", "ltp"):
                col_map[col] = "close"
            elif cl in ("volume", "total traded quantity", "total_traded_quantity"):
                col_map[col] = "volume"
            elif cl in ("date", "ch_timestamp"):
                col_map[col] = "timestamp"

        df = df.rename(columns=col_map)
        # Re-dedup AFTER rename, keeping the first occurrence (the more
        # "canonical" raw name, e.g. CLOSE over LTP, comes first in NSE's
        # column order).
        df = df.loc[:, ~df.columns.duplicated()]

        required = ["open", "high", "low", "close", "volume", "timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            log.error(f"Missing columns for {symbol}: {missing}. Available: {list(df.columns)}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        log.info(f"Fetched {len(df)} daily bars for {symbol} (NSE)")
        return df

    async def fetch_index(
        self, index_name: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_index_sync, index_name, start, end
        )

    def _fetch_index_sync(
        self, index_name: str, start: datetime, end: datetime | None
    ) -> pd.DataFrame:
        from jugaad_data.nse import index_df

        start_date = start.date() if isinstance(start, datetime) else start
        end_date = (end.date() if end else date.today())

        try:
            df = index_df(
                symbol=index_name,
                from_date=start_date,
                to_date=end_date,
            )
        except Exception as e:
            log.error(f"jugaad-data index error for {index_name}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "open" in cl:
                col_map[col] = "open"
            elif "high" in cl:
                col_map[col] = "high"
            elif "low" in cl:
                col_map[col] = "low"
            elif "close" in cl:
                col_map[col] = "close"
            elif "volume" in cl or "shares" in cl:
                col_map[col] = "volume"
            elif "date" in cl:
                col_map[col] = "timestamp"

        df = df.rename(columns=col_map)
        if "volume" not in df.columns:
            df["volume"] = 0

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        log.info(f"Fetched {len(df)} daily bars for {index_name}")
        return df
