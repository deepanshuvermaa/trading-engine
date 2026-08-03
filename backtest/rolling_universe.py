"""NSE-only rolling universe reconstruction for the offline backtest.

Why NSE only: jugaad_data's `full_bhavcopy_save(date, dir)` genuinely
accepts an ARBITRARY historical trading date and returns that day's real
official end-of-day bhavcopy (every listed symbol's OHLC + turnover for
that exact session) — meaning "what would NSE's top-movers basket have
looked like on 2024-11-01" is an honestly answerable question with real
data, not a guess. Neither of the other two markets in this backtest has
an equivalent:

  - Crypto (CoinGecko): the free markets endpoint only returns CURRENT
    24h-volume rankings. There is no free historical "top-volume-by-date"
    endpoint, so there is no way to reconstruct what would have ranked
    top-30 by volume on some date two years ago.
  - US (yfinance): `day_gainers`/`day_losers`/`most_actives` are live
    screeners with no historical-date parameter — they answer "today's"
    movers only.

So crypto/US intentionally stay on the single live-discovery snapshot
applied across the whole backtest window (see backtest/historical_data.py)
— a known, explicitly-flagged data-source limitation, not something faked
here with synthetic historical rankings.

This module reuses `data.universe._fetch_bhavcopy_df` /
`_bhavcopy_top_movers` — the EXACT SAME column-normalisation and
movers/turnover ranking logic the live "today" discovery path uses
(`data.universe.UniverseDiscovery._discover_india_bhavcopy`) — parameterised
by an arbitrary historical date instead of `date.today()`.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from data.universe import _bhavcopy_top_movers, _fetch_bhavcopy_df
from utils.logger import get_logger

log = get_logger("backtest.rolling_universe")


def discover_nse_universe_on(target_date: date, top_n: int = 15) -> list[str]:
    """Reconstruct NSE's top-N movers/most-active basket AS OF `target_date`.

    No-lookahead guarantee: this only ever reads bhavcopy data dated ON or
    BEFORE `target_date` (it walks backward up to 7 calendar days to find
    the nearest actual trading session — weekends/holidays/bhavcopy gaps
    are skipped, never bhavcopy data from a date after `target_date`).
    """
    df, used_date = _fetch_bhavcopy_df(target_date, max_back_days=7, include_anchor=True)
    if df is None or df.empty:
        log.warning(
            f"rolling NSE universe: no bhavcopy found within 7 days on/before "
            f"{target_date.isoformat()} — returning empty cohort for this rebalance point"
        )
        return []
    if used_date != target_date:
        log.info(
            f"rolling NSE universe: {target_date.isoformat()} had no session "
            f"(weekend/holiday/gap) — used nearest PRIOR session {used_date.isoformat()}"
        )

    picks = _bhavcopy_top_movers(df, top_n)
    seen: set[str] = set()
    out: list[str] = []
    for p in picks:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= top_n:
            break
    return out


def monthly_rebalance_dates(lookback_days: int, end_date: date | None = None) -> list[date]:
    """The 1st-of-month candidate date for every calendar month spanned by
    the lookback window (~24 points for a 730-day/2yr window). This is just
    a monthly calendar walk — `discover_nse_universe_on` independently
    walks BACKWARD from whatever date it's given to find the nearest real
    bhavcopy session, so an exact NSE trading-holiday calendar isn't needed
    here; weekends are nudged forward only so the anchor date itself is a
    weekday (a minor convenience, not load-bearing for correctness)."""
    end_date = end_date or date.today()
    start_date = end_date - timedelta(days=lookback_days)

    dates: list[date] = []
    y, m = start_date.year, start_date.month
    while True:
        d = date(y, m, 1)
        if d > end_date:
            break
        while d.weekday() >= 5:  # nudge weekend-1sts to the following Monday
            d += timedelta(days=1)
        if start_date <= d <= end_date:
            dates.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def build_rolling_nse_universe(
    lookback_days: int, top_n: int = 15, end_date: date | None = None
) -> dict[str, Any]:
    """Walk the monthly rebalance schedule, reconstruct that month's real
    NSE top-N basket at each point, and return:
      - rebalance_dates: the ~24 monthly anchor dates (ISO strings)
      - cohorts: {iso_date -> [symbols discovered AS OF that date]}
      - union_symbols: de-duplicated union of every symbol discovered at
        ANY rebalance point (this is what gets fetched full 2yr OHLCV —
        a symbol found in month 3 still needs bars from months 1-2 in
        case some other logic needs its earlier history, even though it
        won't be *scoreable* before its own discovery date; see
        `symbols_live_as_of`).
    """
    rebalance_dates = monthly_rebalance_dates(lookback_days, end_date)
    cohorts: dict[str, list[str]] = {}
    for d in rebalance_dates:
        symbols = discover_nse_universe_on(d, top_n=top_n)
        cohorts[d.isoformat()] = symbols
        log.info(f"rolling NSE universe: {d.isoformat()} -> {len(symbols)} symbols: {symbols}")

    union: list[str] = []
    seen: set[str] = set()
    for syms in cohorts.values():
        for s in syms:
            if s not in seen:
                seen.add(s)
                union.append(s)

    log.info(
        f"rolling NSE universe: {len(rebalance_dates)} monthly rebalance points, "
        f"{len(union)} unique symbols discovered across the whole window"
    )
    return {
        "rebalance_dates": [d.isoformat() for d in rebalance_dates],
        "cohorts": cohorts,
        "union_symbols": union,
    }


def symbols_live_as_of(cohorts: dict[str, list[str]], sim_date: date) -> list[str]:
    """Given a simulated (historical) date, return the symbol cohort from
    that date's MOST RECENT rebalance point — i.e. "what was the discovered
    universe as of this point in simulated time". No-lookahead: only
    rebalance dates <= sim_date are ever considered; if sim_date precedes
    the first rebalance point, returns an empty list (nothing was
    "on the radar" yet)."""
    applicable = [d for d in cohorts if date.fromisoformat(d) <= sim_date]
    if not applicable:
        return []
    latest = max(applicable)
    return cohorts[latest]
