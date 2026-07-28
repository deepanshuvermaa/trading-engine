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


class UniverseDiscovery:
    """Discover the day's most tradeable symbols across all three markets."""

    def __init__(self, cache_minutes: int = 30):
        self.cache_minutes = cache_minutes
        self._cache: dict[str, Any] | None = None
        self._fetched_at: datetime | None = None

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
        india_task = loop.run_in_executor(None, self._discover_india)
        crypto_task = self._discover_crypto()

        us, india, crypto = await asyncio.gather(
            us_task, india_task, crypto_task, return_exceptions=True
        )

        result: dict[str, Any] = {"sources": {}}
        for market, symbols, limit in (
            ("us", us, US_LIMIT),
            ("india", india, INDIA_LIMIT),
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

    # ── India: NSE bhavcopy top movers via jugaad_data ──────────

    @staticmethod
    def _discover_india() -> list[str]:
        import pandas as pd
        from jugaad_data.nse import full_bhavcopy_save

        tmpdir = Path(tempfile.mkdtemp(prefix="nse_bhav_"))
        df = None
        # Walk back up to 7 calendar days to find the latest trading session
        for back in range(1, 8):
            d = date.today() - timedelta(days=back)
            if d.weekday() >= 5:  # skip weekends
                continue
            try:
                path = full_bhavcopy_save(d, str(tmpdir))
                if path is None:
                    candidates = list(tmpdir.glob("*.csv"))
                    path = str(candidates[0]) if candidates else None
                if path:
                    df = pd.read_csv(path)
                    break
            except Exception as e:
                log.debug(f"NSE bhavcopy unavailable for {d}: {e}")
        if df is None or df.empty:
            raise RuntimeError("no NSE bhavcopy found in last 7 days")

        # Normalise column names / string values (NSE pads with spaces)
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

        df = df.copy()
        for c in (close_c, prev_c, turn_c, qty_c):
            if c:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce",
                )

        picks: list[str] = []
        # Half the list: biggest absolute % movers (with real liquidity)
        if close_c and prev_c:
            liquid = df
            if turn_c:
                liquid = df[df[turn_c] >= df[turn_c].quantile(0.5)]
            movers = liquid.assign(
                _chg=((liquid[close_c] - liquid[prev_c]) / liquid[prev_c]).abs()
            ).nlargest(INDIA_LIMIT // 2, "_chg")
            picks += movers["SYMBOL"].tolist()
        # Other half: highest turnover (most active)
        rank_c = turn_c or qty_c
        if rank_c:
            active = df.nlargest(INDIA_LIMIT, rank_c)
            picks += active["SYMBOL"].tolist()

        return [p for p in picks if p and p == p]

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
