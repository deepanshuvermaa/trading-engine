"""Trend-following strategy — EMA crossover + ADX filter + ATR stops."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from data.models import Signal, TradeSignal
from strategy.base import StrategyModule


class TrendFollower(StrategyModule):
    name = "trend_follower"
    description = "EMA crossover trend-following with ADX filter and ATR-based stops"

    def default_params(self) -> dict[str, Any]:
        return {
            "ema_fast": 9,
            "ema_slow": 21,
            "adx_threshold": 25.0,
            "atr_sl_multiplier": 2.0,
            "atr_tp_multiplier": 3.0,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "min_volume_ratio": 0.8,
        }

    def get_param_ranges(self) -> dict[str, tuple[float, float, float]]:
        return {
            "ema_fast": (5, 20, 1),
            "ema_slow": (15, 50, 1),
            "adx_threshold": (15, 40, 5),
            "atr_sl_multiplier": (1.0, 4.0, 0.5),
            "atr_tp_multiplier": (1.5, 6.0, 0.5),
            "rsi_overbought": (65, 85, 5),
            "rsi_oversold": (15, 35, 5),
            "min_volume_ratio": (0.5, 1.5, 0.1),
        }

    def generate_signals(self, df: pd.DataFrame) -> list[TradeSignal]:
        p = self.params
        signals = []

        # Need indicators pre-computed
        required = ["ema_fast", "ema_slow", "adx", "atr", "rsi", "vol_ratio"]
        if not all(col in df.columns for col in required):
            return signals

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            ts = df.index[i]

            if pd.isna(row["adx"]) or pd.isna(row["atr"]):
                continue

            # Skip low-volume environments
            if row["vol_ratio"] < p["min_volume_ratio"]:
                continue

            # ADX filter — only trade in trending markets
            if row["adx"] < p["adx_threshold"]:
                continue

            # Bullish crossover
            if (prev["ema_fast"] <= prev["ema_slow"] and
                    row["ema_fast"] > row["ema_slow"] and
                    row["rsi"] < p["rsi_overbought"]):

                confidence = min(1.0, (row["adx"] - p["adx_threshold"]) / 30 * 0.5 + 0.5)
                signals.append(TradeSignal(
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    symbol=row.get("symbol", df.attrs.get("symbol", "")),
                    signal=Signal.BUY,
                    confidence=confidence,
                    entry_price=row["close"],
                    stop_loss=row["close"] - p["atr_sl_multiplier"] * row["atr"],
                    take_profit=row["close"] + p["atr_tp_multiplier"] * row["atr"],
                    module_name=self.name,
                    reasoning=f"EMA {p['ema_fast']}/{p['ema_slow']} bullish cross, ADX={row['adx']:.1f}, RSI={row['rsi']:.1f}",
                ))

            # Bearish crossover
            elif (prev["ema_fast"] >= prev["ema_slow"] and
                  row["ema_fast"] < row["ema_slow"] and
                  row["rsi"] > p["rsi_oversold"]):

                confidence = min(1.0, (row["adx"] - p["adx_threshold"]) / 30 * 0.5 + 0.5)
                signals.append(TradeSignal(
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    symbol=row.get("symbol", df.attrs.get("symbol", "")),
                    signal=Signal.SELL,
                    confidence=confidence,
                    entry_price=row["close"],
                    stop_loss=row["close"] + p["atr_sl_multiplier"] * row["atr"],
                    take_profit=row["close"] - p["atr_tp_multiplier"] * row["atr"],
                    module_name=self.name,
                    reasoning=f"EMA {p['ema_fast']}/{p['ema_slow']} bearish cross, ADX={row['adx']:.1f}, RSI={row['rsi']:.1f}",
                ))

        return signals
