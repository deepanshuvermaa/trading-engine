"""Market structure analysis — support/resistance, order blocks, trend breaks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def swing_highs_lows(
    high: pd.Series, low: pd.Series, lookback: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Detect swing highs and swing lows using rolling window."""
    swing_high = high[
        (high == high.rolling(2 * lookback + 1, center=True).max())
    ]
    swing_low = low[
        (low == low.rolling(2 * lookback + 1, center=True).min())
    ]
    return swing_high, swing_low


def support_resistance_levels(
    high: pd.Series, low: pd.Series, lookback: int = 5, num_levels: int = 5
) -> tuple[list[float], list[float]]:
    """Extract key S/R levels from swing points."""
    sh, sl = swing_highs_lows(high, low, lookback)
    resistance = sorted(sh.dropna().unique(), reverse=True)[:num_levels]
    support = sorted(sl.dropna().unique())[:num_levels]
    return list(support), list(resistance)


def detect_order_blocks(
    df: pd.DataFrame, lookback: int = 10
) -> pd.DataFrame:
    """
    Detect bullish and bearish order blocks.
    Bullish OB: last bearish candle before a strong bullish move.
    Bearish OB: last bullish candle before a strong bearish move.
    """
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    bearish_candle = close < open_
    bullish_candle = close > open_

    # Strong move = price moves > 2x average candle range
    candle_range = (high - low).rolling(lookback).mean()
    strong_up = (close - open_) > 2 * candle_range
    strong_down = (open_ - close) > 2 * candle_range

    bullish_ob = bearish_candle & strong_up.shift(-1)
    bearish_ob = bullish_candle & strong_down.shift(-1)

    result = pd.DataFrame(index=df.index)
    result["bullish_ob"] = bullish_ob.astype(int)
    result["bearish_ob"] = bearish_ob.astype(int)
    result["ob_high"] = np.where(bullish_ob, high, np.where(bearish_ob, high, np.nan))
    result["ob_low"] = np.where(bullish_ob, low, np.where(bearish_ob, low, np.nan))
    return result


def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap (FVG): gap between candle N-2 high/low and candle N high/low.
    Bullish FVG: candle[i].low > candle[i-2].high
    Bearish FVG: candle[i].high < candle[i-2].low
    """
    high = df["high"]
    low = df["low"]

    bullish_fvg = low > high.shift(2)
    bearish_fvg = high < low.shift(2)

    result = pd.DataFrame(index=df.index)
    result["bullish_fvg"] = bullish_fvg.astype(int)
    result["bearish_fvg"] = bearish_fvg.astype(int)
    result["fvg_top"] = np.where(bullish_fvg, low, np.where(bearish_fvg, low.shift(2), np.nan))
    result["fvg_bottom"] = np.where(bullish_fvg, high.shift(2), np.where(bearish_fvg, high, np.nan))
    return result


def market_structure_break(
    high: pd.Series, low: pd.Series, lookback: int = 5
) -> pd.Series:
    """
    Detect market structure breaks (MSB).
    Bullish MSB: price breaks above recent swing high.
    Bearish MSB: price breaks below recent swing low.
    Returns: Series with 1 (bullish break), -1 (bearish break), 0 (no break).
    """
    swing_high = high.rolling(2 * lookback + 1, center=True).max()
    swing_low = low.rolling(2 * lookback + 1, center=True).min()

    prev_swing_high = swing_high.shift(lookback)
    prev_swing_low = swing_low.shift(lookback)

    bullish_break = (high > prev_swing_high) & (high.shift(1) <= prev_swing_high.shift(1))
    bearish_break = (low < prev_swing_low) & (low.shift(1) >= prev_swing_low.shift(1))

    msb = pd.Series(0, index=high.index)
    msb[bullish_break] = 1
    msb[bearish_break] = -1
    return msb


def trend_regime(df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.Series:
    """
    Classify market regime: TRENDING, RANGING, or VOLATILE.
    Uses ADX + ATR ratio.
    """
    from indicators.technical import adx, atr

    adx_val = adx(df["high"], df["low"], df["close"])
    atr_val = atr(df["high"], df["low"], df["close"])
    atr_pct = atr_val / df["close"] * 100

    regime = pd.Series("RANGING", index=df.index)
    regime[adx_val > adx_threshold] = "TRENDING"
    regime[(adx_val <= adx_threshold) & (atr_pct > atr_pct.rolling(50).mean() * 1.5)] = "VOLATILE"
    return regime


def compute_structure(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Attach all structural analysis to DataFrame."""
    out = df.copy()
    ob = detect_order_blocks(df, lookback)
    fvg = detect_fair_value_gaps(df)
    out["msb"] = market_structure_break(df["high"], df["low"], lookback)
    out["regime"] = trend_regime(df)
    out = pd.concat([out, ob, fvg], axis=1)
    return out
