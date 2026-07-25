"""Base class for all data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.models import Market


class DataProvider(ABC):
    """Abstract interface every data source must implement."""

    market: Market

    def __init__(self, cache_dir: str = "./data/storage"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data. Returns DataFrame with columns: open, high, low, close, volume."""
        ...

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(" ", "_")
        return self.cache_dir / self.market.value / f"{safe_symbol}_{timeframe}.parquet"

    def save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        path = self._cache_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)

    def load_from_cache(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return None
        table = pq.read_table(path)
        df = table.to_pandas()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        return df

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch with caching. Loads cache first, fetches only missing data."""
        cached = self.load_from_cache(symbol, timeframe) if use_cache else None
        if cached is not None and len(cached) > 0:
            # Normalize timezone awareness for comparison
            last_cached = cached.index.max()
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) if end else None

            if last_cached.tzinfo is not None:
                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize(last_cached.tzinfo)
                if end_ts is not None and end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize(last_cached.tzinfo)
            else:
                if start_ts.tzinfo is not None:
                    start_ts = start_ts.tz_localize(None)
                if end_ts is not None and end_ts.tzinfo is not None:
                    end_ts = end_ts.tz_localize(None)

            if end_ts and last_cached >= end_ts:
                mask = (cached.index >= start_ts)
                if end_ts:
                    mask &= (cached.index <= end_ts)
                return cached[mask]
            # Fetch only new data after cache
            new_start = last_cached.to_pydatetime()
            new_data = await self.fetch_ohlcv(symbol, timeframe, new_start, end)
            if len(new_data) > 0:
                combined = pd.concat([cached, new_data])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                self.save_to_cache(combined, symbol, timeframe)
                mask = (combined.index >= start_ts)
                if end_ts:
                    mask &= (combined.index <= end_ts)
                return combined[mask]
            return cached

        df = await self.fetch_ohlcv(symbol, timeframe, start, end)
        if len(df) > 0:
            self.save_to_cache(df, symbol, timeframe)
        return df
