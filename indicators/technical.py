"""Technical indicators — TA-Lib wrapper with fallback to pure pandas."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if HAS_TALIB:
        m, s, h = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return (
            pd.Series(m, index=close.index),
            pd.Series(s, index=close.index),
            pd.Series(h, index=close.index),
        )
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if HAS_TALIB:
        u, m, l = talib.BBANDS(close.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return (
            pd.Series(u, index=close.index),
            pd.Series(m, index=close.index),
            pd.Series(l, index=close.index),
        )
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.EMA(close.values, timeperiod=period), index=close.index)
    return close.ewm(span=period, adjust=False).mean()


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.SMA(close.values, timeperiod=period), index=close.index)
    return close.rolling(period).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
    # Simplified ADX without TA-Lib
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_vals = atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_vals)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_vals)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.rolling(period).mean()


def stochastic_rsi(close: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> tuple[pd.Series, pd.Series]:
    rsi_vals = rsi(close, period)
    stoch_k = (
        (rsi_vals - rsi_vals.rolling(period).min())
        / (rsi_vals.rolling(period).max() - rsi_vals.rolling(period).min())
    ) * 100
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def volume_profile(volume: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling volume relative to average — spike detection."""
    avg_vol = volume.rolling(period).mean()
    return volume / avg_vol.replace(0, np.nan)


def compute_all(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Compute all indicators and attach to DataFrame. Non-destructive."""
    p = params or {}
    out = df.copy()

    out["rsi"] = rsi(df["close"], p.get("rsi_period", 14))
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(
        df["close"], p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_signal", 9)
    )
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger_bands(
        df["close"], p.get("bb_period", 20), p.get("bb_std", 2.0)
    )
    out["atr"] = atr(df["high"], df["low"], df["close"], p.get("atr_period", 14))
    out["ema_fast"] = ema(df["close"], p.get("ema_fast", 9))
    out["ema_slow"] = ema(df["close"], p.get("ema_slow", 21))
    out["sma_50"] = sma(df["close"], 50)
    out["sma_200"] = sma(df["close"], 200)
    out["adx"] = adx(df["high"], df["low"], df["close"], p.get("adx_period", 14))
    out["vol_ratio"] = volume_profile(df["volume"], df["close"], p.get("vol_period", 20))

    return out
