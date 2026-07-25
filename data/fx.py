"""Lightweight FX module — convert native-currency amounts to true USD.

NSE equities are quoted in INR (e.g. Rs.854) while the engine tracks equity in
USD ($100). Sizing "$10 notional" against a raw rupee price and booking P&L in
rupee-points labelled as dollars overstates dollar risk/return by the USDINR
rate (~83x). This module gives a cached USDINR rate so the engine can size and
mark INR positions in real USD.

- usd_rate(currency): units of `currency` per 1 USD (INR/USD). USD -> 1.0.
- to_usd(amount, currency): convert a native amount into USD.

The rate is fetched live via yfinance ("INR=X" / "USDINR=X"), cached for 6h.
If every fetch path fails a hard-coded fallback constant is used and logged, so
the engine never blocks on the FX wire.
"""

from __future__ import annotations

import threading
import time

from utils.logger import get_logger

log = get_logger("data.fx")

# 6-hour cache — FX barely moves intraday and we never want to hammer yfinance.
_CACHE_TTL_SECONDS = 6 * 3600

# Hard fallbacks (native units per 1 USD) used ONLY when the live fetch fails.
# Logged loudly whenever they are used so the source is never ambiguous.
_FALLBACK_RATES: dict[str, float] = {
    "INR": 83.0,
}

_CACHE: dict[str, tuple[float, float, str]] = {}  # currency -> (rate, fetched_at, source)
_LOCK = threading.Lock()

# yfinance tickers to try, in order, per currency.
_FX_TICKERS: dict[str, tuple[str, ...]] = {
    "INR": ("INR=X", "USDINR=X"),
}


def _fetch_rate(currency: str) -> tuple[float, str]:
    """Return (rate, source) — native units per USD. Never raises."""
    for ticker in _FX_TICKERS.get(currency, ()):  # empty tuple -> straight to fallback
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            px = None
            try:
                px = t.fast_info["last_price"]
            except Exception:
                px = None
            if not px:
                hist = t.history(period="5d")
                if len(hist):
                    px = float(hist["Close"].iloc[-1])
            if px and float(px) > 0:
                rate = float(px)
                log.info(f"FX: USD{currency} live rate {rate:.4f} via {ticker}")
                return rate, f"live:{ticker}"
        except Exception as e:  # yfinance flaky / offline — try next, then fallback
            log.warning(f"FX: fetch {ticker} failed: {e}")

    fb = _FALLBACK_RATES.get(currency, 1.0)
    log.warning(f"FX: USD{currency} live fetch failed — using FALLBACK constant {fb}")
    return fb, "fallback"


def usd_rate(currency: str) -> float:
    """Native units of `currency` per 1 USD (cached 6h). USD -> 1.0."""
    cur = (currency or "USD").upper().strip()
    if cur in ("USD", ""):
        return 1.0

    now = time.time()
    with _LOCK:
        cached = _CACHE.get(cur)
        if cached is not None and now - cached[1] < _CACHE_TTL_SECONDS:
            return cached[0]

    rate, source = _fetch_rate(cur)
    with _LOCK:
        _CACHE[cur] = (rate, now, source)
    return rate


def rate_source(currency: str) -> str:
    """Provenance of the currently cached rate: 'live:<ticker>' or 'fallback'."""
    cur = (currency or "USD").upper().strip()
    if cur in ("USD", ""):
        return "identity"
    cached = _CACHE.get(cur)
    return cached[2] if cached else "unfetched"


def to_usd(amount: float, currency: str) -> float:
    """Convert a native-currency amount into USD using the cached rate."""
    if amount is None:
        return amount
    r = usd_rate(currency)
    return float(amount) / r if r else float(amount)
