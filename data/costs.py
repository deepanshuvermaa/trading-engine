"""Per-market transaction-cost model (2026), env-overridable.

Costs differ sharply by venue and — critically for a small account — the Indian
flat brokerage makes tiny NSE orders unviable. This module is the single source
of truth used by BOTH the cost-aware entry filter and realized/unrealized P&L.

  US equities  : commission-free (Alpaca); slippage ~0.03%        -> RT ~0.06%
  Crypto       : 0.10%/side commission; slippage ~0.05%           -> RT ~0.30%
  Indian equity: brokerage = per-order MINIMUM Rs.20 OR 0.03% of
                 turnover, whichever is LARGER (the Rs.20 floor is what bites
                 small orders), + ~0.12% statutory round-trip
                 (STT/GST/stamp/exchange/DP) + slippage ~0.05%.

The Rs.20 per-order floor dominates small notionals: on a $10 (~Rs.830)
position it is ~2.4%/side, ~4.9% round trip — so such trades correctly fail the
3x cost-edge floor, while US (~0.06%) and crypto (~0.30%) setups pass normally.

All figures are returned in TRUE USD. `fx_rate` is native units per USD
(INR/USD ~83); USD markets pass fx_rate=1.0.
"""

from __future__ import annotations

import os


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# market -> cost parameters. commission_pct/slippage_pct/tax_pct are % of
# notional; flat_native is an absolute per-order minimum brokerage in the
# market's native currency; flat_pct is the alternative per-order brokerage rate
# (%), and the LARGER of the two is charged per order.
COST_MODEL: dict[str, dict[str, float]] = {
    "us": {
        "commission_pct": _envf("US_COMMISSION_PCT", 0.0),
        "slippage_pct": _envf("US_SLIPPAGE_PCT", 0.03),
        "flat_native": _envf("US_FLAT_NATIVE", 0.0),
        "flat_pct": _envf("US_FLAT_PCT", 0.0),
        "tax_pct": _envf("US_TAX_PCT", 0.0),
    },
    "crypto": {
        "commission_pct": _envf("CRYPTO_COMMISSION_PCT", 0.10),
        "slippage_pct": _envf("CRYPTO_SLIPPAGE_PCT", 0.05),
        "flat_native": _envf("CRYPTO_FLAT_NATIVE", 0.0),
        "flat_pct": _envf("CRYPTO_FLAT_PCT", 0.0),
        "tax_pct": _envf("CRYPTO_TAX_PCT", 0.0),
    },
    "india": {
        "commission_pct": _envf("INDIA_COMMISSION_PCT", 0.0),
        "slippage_pct": _envf("INDIA_SLIPPAGE_PCT", 0.05),
        "flat_native": _envf("INDIA_FLAT_NATIVE", 20.0),   # Rs.20 per-order floor
        "flat_pct": _envf("INDIA_FLAT_PCT", 0.03),         # or 0.03% of turnover
        "tax_pct": _envf("INDIA_TAX_PCT", 0.12),           # STT+GST+stamp+exch+DP (round-trip)
    },
}


def _model(market: str) -> dict[str, float]:
    return COST_MODEL.get(market, COST_MODEL["crypto"])


def order_cost_usd(market: str, notional_usd: float, fx_rate: float = 1.0) -> float:
    """Cost of ONE order (entry OR exit) in TRUE USD:
    brokerage + slippage + half of the round-trip statutory tax."""
    if notional_usd <= 0:
        return 0.0
    m = _model(market)
    fx = fx_rate or 1.0
    notional_native = notional_usd * fx

    if m["flat_native"] > 0 or m["flat_pct"] > 0:
        # Larger of the Rs.20 floor and the 0.03% rate — the floor bites small
        # orders, the rate bites large ones.
        brokerage_native = max(m["flat_native"], m["flat_pct"] / 100 * notional_native)
        brokerage_usd = brokerage_native / fx
    else:
        brokerage_usd = m["commission_pct"] / 100 * notional_usd

    slippage_usd = m["slippage_pct"] / 100 * notional_usd
    tax_usd = (m["tax_pct"] / 100 * notional_usd) / 2.0  # round-trip tax split per side
    return brokerage_usd + slippage_usd + tax_usd


def round_trip_cost_usd(market: str, notional_usd: float, fx_rate: float = 1.0) -> float:
    """Full round-trip (entry + exit) cost in TRUE USD for this notional."""
    return 2.0 * order_cost_usd(market, notional_usd, fx_rate)


def round_trip_cost_pct(market: str, notional_usd: float, fx_rate: float = 1.0) -> float:
    """Round-trip cost as a % of notional — the number the entry filter gates on."""
    if notional_usd <= 0:
        return 0.0
    return round_trip_cost_usd(market, notional_usd, fx_rate) / notional_usd * 100.0


def cost_breakdown(market: str, notional_usd: float, fx_rate: float = 1.0) -> dict:
    """Detailed per-order + round-trip breakdown for logging/inspection."""
    order = order_cost_usd(market, notional_usd, fx_rate)
    rt = 2.0 * order
    return {
        "market": market,
        "notional_usd": round(notional_usd, 4),
        "order_cost_usd": round(order, 6),
        "round_trip_cost_usd": round(rt, 6),
        "round_trip_cost_pct": round(round_trip_cost_pct(market, notional_usd, fx_rate), 4),
    }
