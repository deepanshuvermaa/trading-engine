"""Macro economic data via FRED (Federal Reserve) + GDELT event queries."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, date
from typing import Any

import httpx
import pandas as pd

from data.ingestion.base import DataProvider
from data.models import Market
from utils.logger import get_logger

log = get_logger("data.macro")

# Key FRED series for macro regime detection
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "us_cpi": "CPIAUCSL",
    "us_10y_yield": "DGS10",
    "us_2y_yield": "DGS2",
    "us_unemployment": "UNRATE",
    "us_gdp": "GDP",
    "m2_money_supply": "M2SL",
    "vix": "VIXCLS",
    "us_dollar_index": "DTWEXBGS",
    "sp500": "SP500",
}


class FREDProvider:
    """Pull macro indicators from FRED. No subclass of DataProvider — different shape."""

    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY", "")

    async def fetch_series(
        self, series_id: str, start: datetime, end: datetime | None = None
    ) -> pd.Series:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_sync, series_id, start, end
        )

    def _fetch_sync(
        self, series_id: str, start: datetime, end: datetime | None
    ) -> pd.Series:
        if not self.api_key:
            log.warning("FRED_API_KEY not set — skipping FRED data")
            return pd.Series(dtype=float)

        from fredapi import Fred
        fred = Fred(api_key=self.api_key)

        try:
            data = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end or date.today(),
            )
            log.info(f"FRED {series_id}: {len(data)} observations")
            return data.dropna()
        except Exception as e:
            log.error(f"FRED error for {series_id}: {e}")
            return pd.Series(dtype=float)

    async def fetch_all_macro(
        self, start: datetime, end: datetime | None = None
    ) -> dict[str, pd.Series]:
        """Fetch all key macro series in parallel."""
        tasks = {
            name: self.fetch_series(series_id, start, end)
            for name, series_id in FRED_SERIES.items()
        }
        results = {}
        for name, coro in tasks.items():
            results[name] = await coro
        return results

    async def get_yield_curve_spread(
        self, start: datetime, end: datetime | None = None
    ) -> pd.Series:
        """10Y - 2Y spread. Negative = inverted yield curve = recession signal."""
        ten_y = await self.fetch_series("DGS10", start, end)
        two_y = await self.fetch_series("DGS2", start, end)
        combined = pd.DataFrame({"10y": ten_y, "2y": two_y}).dropna()
        return combined["10y"] - combined["2y"]


class GDELTProvider:
    """Query GDELT event database for geopolitical event data."""

    BASE_URL = "https://api.gdeltproject.org/api/v2"

    async def search_events(
        self,
        query: str,
        mode: str = "ArtList",  # ArtList, TimelineVol, TimelineTone
        timespan: str = "3months",
        max_records: int = 250,
    ) -> list[dict[str, Any]]:
        """Search GDELT for events matching query."""
        params = {
            "query": query,
            "mode": mode,
            "maxrecords": max_records,
            "timespan": timespan,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(f"{self.BASE_URL}/doc/doc", params=params)
                resp.raise_for_status()
                data = resp.json()
                articles = data.get("articles", [])
                log.info(f"GDELT '{query}': {len(articles)} articles")
                return articles
            except Exception as e:
                log.error(f"GDELT error: {e}")
                return []

    async def get_tone_timeline(
        self, query: str, timespan: str = "1year"
    ) -> pd.DataFrame:
        """Get sentiment tone over time for a query."""
        params = {
            "query": query,
            "mode": "TimelineTone",
            "timespan": timespan,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(f"{self.BASE_URL}/doc/doc", params=params)
                resp.raise_for_status()
                data = resp.json()
                timeline = data.get("timeline", [])
                if not timeline:
                    return pd.DataFrame(columns=["date", "tone"])

                rows = []
                for series in timeline:
                    for point in series.get("data", []):
                        rows.append({
                            "date": point.get("date", ""),
                            "tone": point.get("value", 0.0),
                        })
                df = pd.DataFrame(rows)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                return df
            except Exception as e:
                log.error(f"GDELT timeline error: {e}")
                return pd.DataFrame(columns=["date", "tone"])
