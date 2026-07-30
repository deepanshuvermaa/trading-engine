"""Crypto data via CCXT — Binance default, supports any exchange."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd

from data.models import Market
from data.ingestion.base import DataProvider
from utils.logger import get_logger

log = get_logger("data.crypto")

# CCXT timeframe string → milliseconds
TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000,
}


class CryptoProvider(DataProvider):
    market = Market.CRYPTO

    def __init__(self, exchange_id: str = "binance", cache_dir: str = "./data/storage"):
        super().__init__(cache_dir)
        self.exchange_id = exchange_id
        self._exchange: ccxt.Exchange | None = None

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            cls = getattr(ccxt, self.exchange_id)
            self._exchange = cls({"enableRateLimit": True})
        return self._exchange

    async def close(self):
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    async def fetch_intraday(
        self, symbol: str, interval: str = "15m", lookback_days: int = 30
    ) -> pd.DataFrame:
        """Intraday bars via yfinance (Binance is geo-blocked here, and 15m
        crypto works fine on yfinance). BTC/USDT -> BTC-USD."""
        from data.ingestion.intraday import fetch_yf_intraday

        yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
        if "/" in yf_symbol:  # any other quote — take the base against USD
            yf_symbol = f"{symbol.split('/')[0]}-USD"
        return await fetch_yf_intraday(yf_symbol, interval, lookback_days)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        exchange = await self._get_exchange()
        since_ms = int(start.timestamp() * 1000)
        end_ms = int((end or datetime.now(timezone.utc)).timestamp() * 1000)
        tf_ms = TF_MS.get(timeframe, 3_600_000)
        limit = 1000  # Binance max per request

        all_candles = []
        current = since_ms

        while current < end_ms:
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=current, limit=limit
                )
            except Exception as e:
                log.error(f"CCXT error {symbol} {timeframe}: {e}")
                break

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if last_ts <= current:
                break
            current = last_ts + tf_ms

            # Rate limit courtesy
            await asyncio.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            # Fallback to yfinance for daily data
            if timeframe in ("1d", "1w"):
                log.warning(f"CCXT failed for {symbol}, falling back to yfinance")
                return await self._yfinance_fallback(symbol, timeframe, start, end)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Trim to requested range. `end` may arrive tz-aware (e.g. callers
        # passing datetime.now(timezone.utc), as backtest/historical_data.py
        # does) or tz-naive -- pd.Timestamp(x, tz="UTC") raises
        # "Cannot pass a datetime or Timestamp with tzinfo with the tz
        # parameter" if x is ALREADY tz-aware, which was silently failing
        # every single crypto historical fetch call from that path,
        # including BTC/USDT -- not an IP-block/rate-limit issue at all,
        # a real bug in this file.
        if end:
            end_ts = pd.Timestamp(end)
            end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
            df = df[df.index <= end_ts]

        log.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    async def _yfinance_fallback(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None
    ) -> pd.DataFrame:
        """Fallback to yfinance when exchange APIs are blocked."""
        import yfinance as yf

        # Convert CCXT symbol to yfinance format: BTC/USDT → BTC-USD
        yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
        interval = "1d" if timeframe == "1d" else "1wk"

        loop = asyncio.get_event_loop()
        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            return ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d") if end else None,
                interval=interval,
                auto_adjust=True,
            )

        try:
            df = await loop.run_in_executor(None, _fetch)
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]]
            df.index.name = "timestamp"
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            log.info(f"yfinance fallback: {len(df)} bars for {yf_symbol} {timeframe}")
            return df
        except Exception as e:
            log.error(f"yfinance fallback failed for {yf_symbol}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
