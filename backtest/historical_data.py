"""Historical OHLCV puller for the offline learning-validation backtest.

Reuses the EXACT SAME production data providers the live engine uses — no new
scraping, no new data source:
  - Crypto : data.ingestion.crypto.CryptoProvider   (yfinance-backed daily)
  - US     : data.ingestion.us_equity.USEquityProvider (yfinance)
  - NSE    : data.ingestion.indian_equity.IndianEquityProvider
             (jugaad-data — NSE's own official bhavcopy archive)

Fetched bars are cached to parquet under data/storage/backtest_cache/ (that
whole tree is already covered by the repo's `data/storage/**` gitignore rule)
so repeat runs are fast and auditable — every fetch is logged with the exact
symbol, date range, bar count and provider used.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.ingestion.crypto import CryptoProvider
from data.ingestion.indian_equity import IndianEquityProvider
from data.ingestion.us_equity import USEquityProvider
from utils.logger import get_logger

log = get_logger("backtest.historical_data")

CACHE_DIR = str(Path(__file__).resolve().parent.parent / "data" / "storage" / "backtest_cache")

LOOKBACK_DAYS = 730  # ~2 calendar years

# No hardcoded symbol lists. The basket for every market is discovered live
# by data.universe.UniverseDiscovery -- the SAME dynamic top-movers/
# most-active mechanism the production Scout loop uses to pick what to
# trade. A backtest built on a hand-picked list (blue-chips, or someone's
# guess at "volatile mid-caps") tests the strategy against a tape that
# doesn't match what it actually trades live -- it proves nothing about
# real performance. Sourcing the basket from live discovery means whatever
# symbols get backtested are, by construction, the same kind of names the
# system would genuinely be looking at today.
#
# NSE gets the gold-standard treatment: a TRUE ROLLING universe, reconstructed
# monthly from real historical NSE bhavcopy (backtest/rolling_universe.py) --
# a different, date-correct basket at each of ~24 monthly rebalance points
# across the lookback window, with no lookahead (a rebalance point never
# reads bhavcopy data from after its own date). This is possible for NSE
# ONLY because jugaad_data's full_bhavcopy_save(date, dir) genuinely accepts
# an arbitrary historical date.
#
# Crypto and US remain a SINGLE point-in-time snapshot (today's movers)
# applied across the entire window -- NOT a design choice, a hard data-source
# limitation: CoinGecko's free markets endpoint has no historical
# top-volume-by-date API, and yfinance's day_gainers/day_losers/most_actives
# screeners are live-only with no historical-date parameter. There is no
# free, real data source that answers "what would have been top-movers by
# volume on 2024-11-01" for either market. Flagged honestly here and in the
# generated report, not silently glossed over.
BASKET_SIZE_PER_MARKET = 15


async def _discover_baskets(
    lookback_days: int,
) -> tuple[dict[str, dict[str, tuple[str, str]]], dict]:
    """Crypto/US: TODAY's real top-movers/most-active via the live
    UniverseDiscovery mechanism (single snapshot, applied across the whole
    window -- see the module-level comment above for why).

    NSE: a TRUE rolling reconstruction -- ~24 monthly rebalance points, each
    asking "what would NSE's top movers/most-active basket have looked like
    AS OF this historical date", via backtest.rolling_universe (real
    bhavcopy, no lookahead). The union of everything discovered across all
    rebalance points becomes the NSE fetch list; per-rebalance-point cohorts
    are returned separately so the walk-forward simulation can restrict
    which symbols are actually *scoreable* at any given simulated date.

    Sector/company-name labels aren't available from discovery, so symbols
    are labeled honestly by market/mechanism rather than inventing a sector
    we didn't actually look up.
    """
    from data.universe import UniverseDiscovery
    from backtest.rolling_universe import build_rolling_nse_universe

    uni = await UniverseDiscovery().discover(force=True)
    labels = {"crypto": "Live Universe Discovery — Crypto (24h volume rank; "
                         "single live snapshot, applied across whole window — "
                         "no free historical top-volume-by-date API)",
              "us": "Live Universe Discovery — US (gainers/losers/most-active; "
                    "single live snapshot, applied across whole window — "
                    "yfinance screeners are live-only)"}
    baskets: dict[str, dict[str, tuple[str, str]]] = {}
    for market in ("crypto", "us"):
        symbols = list(uni.get(market, []))[:BASKET_SIZE_PER_MARKET]
        baskets[market] = {s: (s, labels[market]) for s in symbols}

    loop = asyncio.get_event_loop()
    rolling = await loop.run_in_executor(
        None, build_rolling_nse_universe, lookback_days, BASKET_SIZE_PER_MARKET, None)
    india_label = (f"Rolling NSE Universe Reconstruction — "
                   f"{len(rolling['rebalance_dates'])} monthly rebalance points, "
                   f"real bhavcopy, no lookahead (backtest.rolling_universe)")
    baskets["india"] = {s: (s, india_label) for s in rolling["union_symbols"]}

    return baskets, rolling

PROVIDER_LABEL: dict[str, str] = {
    "crypto": "CryptoProvider (Binance via ccxt, yfinance daily fallback)",
    "us": "USEquityProvider (yfinance daily)",
    "india": "IndianEquityProvider (jugaad-data — NSE official bhavcopy archive)",
}


@dataclass
class FetchReport:
    """One line of the data-authenticity trail: exactly what was pulled,
    from where, and over what real calendar range."""

    market: str
    symbol: str
    name: str
    sector: str
    ok: bool
    bars: int = 0
    start: str | None = None
    end: str | None = None
    provider: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "market": self.market, "symbol": self.symbol, "name": self.name,
            "sector": self.sector, "ok": self.ok, "bars": self.bars,
            "start": self.start, "end": self.end, "provider": self.provider,
            "error": self.error,
        }


async def fetch_all(
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[FetchReport], dict]:
    """Fetch 1D OHLCV for the representative baskets via the LIVE providers.

    Returns (data[market][symbol] -> DataFrame, audit reports — one per
    symbol attempted, success or failure, for the authenticity trail,
    rolling_universe — the NSE monthly rebalance-point metadata
    (rebalance_dates/cohorts/union_symbols) from backtest.rolling_universe,
    empty dict values for crypto/US since they're a single snapshot).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    baskets, rolling = await _discover_baskets(lookback_days)
    total_discovered = sum(len(b) for b in baskets.values())
    log.info(f"historical_data: basket discovered live — "
             f"{ {m: len(b) for m, b in baskets.items()} } "
             f"({total_discovered} symbols total, not hardcoded)")

    crypto = CryptoProvider("binance", CACHE_DIR)
    us_eq = USEquityProvider(CACHE_DIR)
    indian_eq = IndianEquityProvider(CACHE_DIR)
    providers = {"crypto": crypto, "us": us_eq, "india": indian_eq}

    data: dict[str, dict[str, pd.DataFrame]] = {"crypto": {}, "us": {}, "india": {}}
    reports: list[FetchReport] = []

    async def fetch_one(market: str, symbol: str, name: str, sector: str):
        try:
            df = await providers[market].get_ohlcv(symbol, "1d", start, end, use_cache=True)
        except Exception as e:  # never let one bad symbol kill the whole pull
            log.error(f"{market}:{symbol} fetch failed: {e}")
            reports.append(FetchReport(market, symbol, name, sector, False,
                                        provider=PROVIDER_LABEL[market], error=str(e)))
            return
        n = 0 if df is None else len(df)
        if df is None or n < 60:
            reports.append(FetchReport(market, symbol, name, sector, False,
                                        provider=PROVIDER_LABEL[market],
                                        error=f"insufficient bars ({n})"))
            return
        # Normalize to tz-naive (US/crypto providers return tz-aware
        # exchange-local indices; NSE via jugaad-data is tz-naive) so every
        # market's daily bars share one comparable calendar downstream, and
        # de-dup/sort in case a stale parquet cache and a fresh fetch land on
        # the same calendar day under different tz representations.
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        data[market][symbol] = df
        reports.append(FetchReport(
            market, symbol, name, sector, True, bars=n,
            start=str(df.index[0].date()), end=str(df.index[-1].date()),
            provider=PROVIDER_LABEL[market]))

    tasks = [
        fetch_one(market, symbol, name, sector)
        for market, basket in baskets.items()
        for symbol, (name, sector) in basket.items()
    ]
    await asyncio.gather(*tasks)

    if hasattr(crypto, "close"):
        try:
            await crypto.close()
        except Exception:
            pass

    ok = sum(1 for r in reports if r.ok)
    log.info(f"historical_data: fetched {ok}/{len(reports)} symbols across "
             f"{len(baskets)} markets, lookback {lookback_days}d "
             f"({start.date()} .. {end.date()})")
    for r in sorted(reports, key=lambda r: (r.market, r.symbol)):
        if r.ok:
            log.info(f"  OK   {r.market:6s} {r.symbol:12s} {r.bars:4d} bars "
                      f"[{r.start} .. {r.end}] via {r.provider}")
        else:
            log.warning(f"  FAIL {r.market:6s} {r.symbol:12s} — {r.error}")

    return data, reports, rolling
