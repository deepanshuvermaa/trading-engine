"""
Dynamic Universe Discovery — no hardcoded symbol lists.

Builds the scan universe fresh each cycle (30-min cached) from live sources:
- US:     yfinance predefined screeners (day_gainers, day_losers, most_actives)
- India:  latest NSE bhavcopy via jugaad_data — top movers by turnover and
          absolute price change
- Crypto: CoinGecko markets ranked by 24h volume, mapped to SYMBOL/USDT

A tiny hardwired default is used ONLY when a discovery source fails outright,
and that fallback is logged loudly.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any

import httpx

from data.ingestion.nse_live import get_nse_live_session
from utils.logger import get_logger

log = get_logger("data.universe")

US_LIMIT = 50
INDIA_LIMIT = 75
CRYPTO_LIMIT = 30

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=volume_desc&per_page={n}&page=1"
)

# Stable/pegged coins — pointless to trade against USDT
STABLE_BASES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS",
                "BUSD", "PYUSD", "USDD", "WBTC", "WETH", "STETH", "WSTETH",
                "WEETH", "CBBTC"}

# Emergency fallbacks — used ONLY when the live discovery source fails
FALLBACK = {
    "crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"],
    "us": ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA"],
    "india": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"],
}


def _normalize_bhavcopy_df(df: Any) -> Any:
    """Column-name/value normalisation shared by every bhavcopy consumer —
    NSE's raw CSV headers/format have changed over the years (spaced-padded
    strings, comma-thousands numerics, CLOSE_PRICE vs CLOSE, etc). Any code
    that reads a bhavcopy DataFrame MUST go through this first so today's
    live discovery and the historical rolling-universe reconstruction
    (backtest/rolling_universe.py) are provably parsing it identically."""
    import pandas as pd

    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    for col in ("SYMBOL", "SERIES"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "SERIES" in df.columns:
        df = df[df["SERIES"] == "EQ"]

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    close_c = col("CLOSE_PRICE", "CLOSE")
    prev_c = col("PREV_CLOSE", "PREVCLOSE")
    turn_c = col("TURNOVER_LACS", "TOTTRDVAL", "TURNOVER")
    qty_c = col("TTL_TRD_QNTY", "TOTTRDQTY")

    for c in (close_c, prev_c, turn_c, qty_c):
        if c:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )

    df.attrs["_close_col"] = close_c
    df.attrs["_prev_col"] = prev_c
    df.attrs["_turnover_col"] = turn_c
    df.attrs["_qty_col"] = qty_c
    return df


def _fetch_bhavcopy_df(
    anchor_date: date, max_back_days: int = 7, include_anchor: bool = False
) -> tuple[Any, date | None]:
    """Fetch+normalise the nearest NSE bhavcopy on/before `anchor_date`.

    `include_anchor=True` tries `anchor_date` itself first, then walks
    backward through `max_back_days` prior calendar days (skipping
    weekends) until a session is found; `include_anchor=False` starts the
    walk at `anchor_date - 1 day` (used by the live/"today" path, since
    today's own bhavcopy isn't published until after the session closes).

    Never looks FORWARD of `anchor_date` — this is the load-bearing
    no-lookahead guarantee for the historical rolling-universe
    reconstruction. Returns (None, None) if nothing is found."""
    import pandas as pd
    from jugaad_data.nse import full_bhavcopy_save

    tmpdir = Path(tempfile.mkdtemp(prefix="nse_bhav_"))
    start_offset = 0 if include_anchor else 1
    for back in range(start_offset, max_back_days + start_offset):
        d = anchor_date - timedelta(days=back)
        if d.weekday() >= 5:  # skip weekends
            continue
        try:
            path = full_bhavcopy_save(d, str(tmpdir))
            if path is None:
                candidates = list(tmpdir.glob("*.csv"))
                path = str(candidates[0]) if candidates else None
            if path:
                raw = pd.read_csv(path)
                return _normalize_bhavcopy_df(raw), d
        except Exception as e:
            log.debug(f"NSE bhavcopy unavailable for {d}: {e}")
    return None, None


def _bhavcopy_top_movers(df: Any, limit: int) -> list[str]:
    """Given an already-normalised bhavcopy DataFrame (see
    `_normalize_bhavcopy_df`), rank symbols the same way both the live
    discovery path and the historical rolling-universe path do: half by
    biggest absolute %-change among liquid names (turnover >= median),
    half by raw turnover/quantity ("most active"). Parameterised by
    `limit` so callers can ask for a small historical basket (backtest,
    top_n=15) or the larger live scan basket (INDIA_LIMIT)."""
    close_c = df.attrs.get("_close_col")
    prev_c = df.attrs.get("_prev_col")
    turn_c = df.attrs.get("_turnover_col")
    qty_c = df.attrs.get("_qty_col")

    picks: list[str] = []
    if close_c and prev_c:
        liquid = df
        if turn_c:
            liquid = df[df[turn_c] >= df[turn_c].quantile(0.5)]
        movers = liquid.assign(
            _chg=((liquid[close_c] - liquid[prev_c]) / liquid[prev_c]).abs()
        ).nlargest(max(limit // 2, 1), "_chg")
        picks += movers["SYMBOL"].tolist()
    rank_c = turn_c or qty_c
    if rank_c:
        active = df.nlargest(limit, rank_c)
        picks += active["SYMBOL"].tolist()

    return [p for p in picks if p and p == p]


class UniverseDiscovery:
    """Discover the day's most tradeable symbols across all three markets."""

    def __init__(self, cache_minutes: int = 30):
        self.cache_minutes = cache_minutes
        self._cache: dict[str, Any] | None = None
        self._fetched_at: datetime | None = None
        # External push override (the local relay bridge — see
        # scripts/nse_local_relay.py). This is TIER 1 of a graceful fallback
        # chain: push override (fresh) -> Railway's own live NSE session ->
        # bhavcopy -> hardcoded 5-name safety net. The relay depends on the
        # operator's machine being on and connected, so it can NEVER be the
        # only path -- if it's stale/absent, every call below just proceeds
        # exactly as it did before this existed.
        self._india_override: list[str] | None = None
        self._india_override_at: datetime | None = None
        self._india_override_ttl_min = 30

    def set_india_override(self, symbols: list[str]) -> None:
        """Called by the /api/universe/push-nse endpoint. Expires on its own
        after _india_override_ttl_min -- no separate cleanup needed, and a
        relay that goes offline mid-session degrades silently to bhavcopy."""
        self._india_override = [s.strip().upper() for s in symbols if s and s.strip()]
        self._india_override_at = datetime.now(timezone.utc)
        # Force the next discover() to re-check india (don't serve a stale
        # cached bhavcopy result for up to cache_minutes after a fresh push).
        self._fetched_at = None

    def _india_override_fresh(self) -> list[str] | None:
        if not self._india_override or not self._india_override_at:
            return None
        age_min = (datetime.now(timezone.utc) - self._india_override_at).total_seconds() / 60.0
        if age_min > self._india_override_ttl_min:
            return None
        return self._india_override

    async def discover(self, force: bool = False) -> dict[str, Any]:
        """Return {"crypto": [...], "us": [...], "india": [...],
        "counts": {...}, "sources": {...}, "updated_at": iso}."""
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._cache is not None
            and self._fetched_at is not None
            and now - self._fetched_at < timedelta(minutes=self.cache_minutes)
        ):
            return self._cache

        loop = asyncio.get_event_loop()
        us_task = loop.run_in_executor(None, self._discover_us)
        india_task = self._discover_india()
        crypto_task = self._discover_crypto()

        us, india_res, crypto = await asyncio.gather(
            us_task, india_task, crypto_task, return_exceptions=True
        )

        result: dict[str, Any] = {"sources": {}}
        for market, symbols, limit in (
            ("us", us, US_LIMIT),
            ("crypto", crypto, CRYPTO_LIMIT),
        ):
            if isinstance(symbols, Exception) or not symbols:
                err = symbols if isinstance(symbols, Exception) else "empty result"
                log.error(
                    f"UNIVERSE FALLBACK [{market}]: live discovery failed "
                    f"({err}) — using minimal default list "
                    f"({len(FALLBACK[market])} symbols)"
                )
                result[market] = list(FALLBACK[market])
                result["sources"][market] = "FALLBACK_DEFAULT"
            else:
                result[market] = self._dedupe(symbols)[:limit]
                result["sources"][market] = "live"

        # India: (symbols, source_label) — live NSE session first, bhavcopy
        # fallback carries the day when NSE blocks/rate-limits us (see
        # _discover_india's own auditable logging of which path fired).
        if isinstance(india_res, Exception) or not india_res or not india_res[0]:
            err = india_res if isinstance(india_res, Exception) else "empty result"
            log.error(
                f"UNIVERSE FALLBACK [india]: live discovery failed "
                f"({err}) — using minimal default list "
                f"({len(FALLBACK['india'])} symbols)"
            )
            result["india"] = list(FALLBACK["india"])
            result["sources"]["india"] = "FALLBACK_DEFAULT"
        else:
            india_symbols, india_source = india_res
            result["india"] = self._dedupe(india_symbols)[:INDIA_LIMIT]
            result["sources"]["india"] = india_source

        result["counts"] = {m: len(result[m]) for m in ("us", "india", "crypto")}
        result["total"] = sum(result["counts"].values())
        result["updated_at"] = now.isoformat()

        self._cache = result
        self._fetched_at = now
        log.info(
            f"Universe: {result['total']} securities — "
            f"{result['counts']['us']} US, {result['counts']['india']} NSE, "
            f"{result['counts']['crypto']} crypto "
            f"(sources: {result['sources']})"
        )
        return result

    # ── US: yfinance predefined screeners ───────────────────────

    @staticmethod
    def _discover_us() -> list[str]:
        import yfinance as yf

        symbols: list[str] = []
        for screener in ("day_gainers", "day_losers", "most_actives"):
            try:
                quotes: list[dict] = []
                if hasattr(yf, "screen"):
                    # yfinance >= 0.2.5x
                    try:
                        resp = yf.screen(screener, size=25)
                    except TypeError:
                        resp = yf.screen(screener)
                    if isinstance(resp, dict):
                        quotes = resp.get("quotes", [])
                elif hasattr(yf, "Screener"):
                    s = yf.Screener()
                    s.set_predefined_body(screener)
                    quotes = (s.response or {}).get("quotes", [])
                for q in quotes[:25]:
                    sym = (q.get("symbol") or "").strip().upper()
                    if sym and "=" not in sym and "^" not in sym:
                        symbols.append(sym)
            except Exception as e:
                log.warning(f"US screener '{screener}' failed: {e}")
        return symbols

    # ── India: live NSE session FIRST, bhavcopy fallback ─────────

    async def _discover_india(self) -> tuple[list[str], str]:
        """Fallback chain, richest-to-safest:
          1. Local relay push override (fresh, <=30min old) -- a machine off
             NSE's cloud-IP blocklist did the live-session fetch and pushed
             the result here. Only source that can see TODAY's real movers
             from Railway.
          2. Railway's own live NSE session attempt -- usually 403s (NSE
             blocks known cloud-provider IP ranges) but costs nothing to try,
             and may work if that changes.
          3. Bhavcopy -- yesterday's official NSE settlement file. Always
             available on trading days, the load-bearing safety net.
          4. Hardcoded 5-name default (FALLBACK) -- only if bhavcopy itself
             is unreachable.
        Every tier is independent; if 1 is stale/absent, 2-4 behave EXACTLY
        as they did before the relay existed. Returns (symbols, source_label)
        so the caller can log/audit which tier actually fired this cycle."""
        override = self._india_override_fresh()
        if override:
            age_min = (datetime.now(timezone.utc) - self._india_override_at).total_seconds() / 60.0
            log.info(f"India universe: local relay push override — "
                     f"{len(override)} symbols ({age_min:.0f}min old)")
            return override, "local_relay_live"

        try:
            live_symbols = await self._discover_india_live()
        except Exception as e:
            log.warning(f"India universe: live NSE session errored ({e})")
            live_symbols = []

        if live_symbols:
            log.info(f"India universe: live NSE session — "
                     f"{len(live_symbols)} symbols")
            return live_symbols, "live_nse_session"

        log.warning("India universe: live NSE session returned nothing "
                    "usable — falling back to bhavcopy (yesterday's data)")
        loop = asyncio.get_event_loop()
        try:
            bhav_symbols = await loop.run_in_executor(
                None, self._discover_india_bhavcopy)
        except Exception as e:
            log.error(f"India universe: bhavcopy fallback also failed ({e})")
            return [], "bhavcopy_fallback"
        log.info(f"India universe: bhavcopy fallback (yesterday's data) — "
                 f"{len(bhav_symbols)} symbols")
        return bhav_symbols, "bhavcopy_fallback"

    @staticmethod
    async def _discover_india_live() -> list[str]:
        """Today's live gainers/losers + most-active via NSELiveSession.
        Returns [] (never raises) if NSE's live endpoints don't cooperate —
        the caller falls back to bhavcopy."""
        session = get_nse_live_session()
        picks: list[str] = []
        try:
            gl = await session.get_live_gainers_losers()
            for side in ("gainers", "losers"):
                picks += [r["symbol"] for r in gl.get(side, [])]
        except Exception as e:
            log.debug(f"India live gainers/losers unavailable: {e}")
        try:
            active = await session.get_live_most_active()
            picks += [r["symbol"] for r in active]
        except Exception as e:
            log.debug(f"India live most-active unavailable: {e}")
        if not picks:
            try:
                variations = await session.get_index_variations("NIFTY 500")
                # Rank by absolute live %-change — the day's real movers.
                variations.sort(key=lambda r: abs(r.get("pct_change", 0)), reverse=True)
                picks += [r["symbol"] for r in variations[:INDIA_LIMIT]]
            except Exception as e:
                log.debug(f"India live index variations unavailable: {e}")
        return [p for p in picks if p]

    @staticmethod
    def _discover_india_bhavcopy() -> list[str]:
        # Today's picks: walk BACKWARD from yesterday (today's own bhavcopy
        # isn't published until after the session closes, so `include_anchor`
        # is deliberately False here — see backtest.rolling_universe for the
        # historical-reconstruction variant that DOES anchor on its own date).
        df, _used_date = _fetch_bhavcopy_df(date.today(), max_back_days=7, include_anchor=False)
        if df is None or df.empty:
            raise RuntimeError("no NSE bhavcopy found in last 7 days")
        return _bhavcopy_top_movers(df, INDIA_LIMIT)

    # ── Crypto: CoinGecko by 24h volume ─────────────────────────

    @staticmethod
    async def _discover_crypto() -> list[str]:
        url = COINGECKO_URL.format(n=CRYPTO_LIMIT + len(STABLE_BASES))
        async with httpx.AsyncClient(
            timeout=20, headers={"User-Agent": "Mozilla/5.0"}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            coins = resp.json()

        pairs = []
        for c in coins:
            base = (c.get("symbol") or "").upper().strip()
            if base and base not in STABLE_BASES:
                pairs.append(f"{base}/USDT")
        return pairs

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out
