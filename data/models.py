"""Core data models used across the entire engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel


class Market(str, Enum):
    CRYPTO = "crypto"
    INDIAN_EQUITY = "indian_equity"
    US_EQUITY = "us_equity"
    FOREX = "forex"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ModuleStatus(str, Enum):
    TRIAL = "TRIAL"
    ACTIVE = "ACTIVE"
    PROBATION = "PROBATION"
    DEACTIVATED = "DEACTIVATED"
    TERMINATED = "TERMINATED"


class OHLCV(BaseModel):
    """Single candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str

    class Config:
        frozen = True


class TradeSignal(BaseModel):
    """Output from a strategy module."""
    timestamp: datetime
    symbol: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    module_name: str
    reasoning: str = ""


class BacktestResult(BaseModel):
    """Metrics from a single backtest run."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_return_pct: float = 0.0
    avg_trade_return_pct: float = 0.0
    calmar_ratio: float = 0.0

    @property
    def composite_score(self) -> float:
        """Single score combining key metrics. Higher = better."""
        if self.total_trades < 5:
            return 0.0
        sharpe_component = max(0, self.sharpe_ratio) * 0.35
        winrate_component = self.win_rate * 0.20
        pf_component = min(self.profit_factor, 3.0) / 3.0 * 0.20
        dd_component = max(0, 1.0 - abs(self.max_drawdown_pct) / 10.0) * 0.25
        return (sharpe_component + winrate_component + pf_component + dd_component) * 10


def ohlcv_to_dataframe(candles: list[OHLCV]) -> pd.DataFrame:
    """Convert list of OHLCV to a pandas DataFrame indexed by timestamp."""
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame([c.model_dump() for c in candles])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def dataframe_to_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> list[OHLCV]:
    """Convert DataFrame back to OHLCV list."""
    records = []
    for ts, row in df.iterrows():
        records.append(OHLCV(
            timestamp=ts.to_pydatetime(),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            symbol=symbol,
            timeframe=timeframe,
        ))
    return records
