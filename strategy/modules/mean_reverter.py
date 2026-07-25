"""Mean-reversion strategy — Bollinger Band extremes + RSI divergence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data.models import Signal, TradeSignal
from strategy.base import StrategyModule


class MeanReverter(StrategyModule):
    name = "mean_reverter"
    description = "Bollinger Band mean-reversion with RSI extremes and volume confirmation"

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "adx_max": 25.0,  # Only trade in ranging markets
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 1.0,  # Target the mean (middle band)
            "min_volume_ratio": 1.2,  # Need above-average volume for reversals
        }

    def get_param_ranges(self) -> dict[str, tuple[float, float, float]]:
        return {
            "bb_period": (10, 30, 5),
            "bb_std": (1.5, 3.0, 0.5),
            "rsi_period": (7, 21, 7),
            "rsi_oversold": (20, 40, 5),
            "rsi_overbought": (60, 80, 5),
            "adx_max": (15, 35, 5),
            "atr_sl_multiplier": (1.0, 3.0, 0.5),
            "min_volume_ratio": (0.8, 2.0, 0.2),
        }

    def generate_signals(self, df: pd.DataFrame) -> list[TradeSignal]:
        p = self.params
        signals = []

        required = ["bb_upper", "bb_lower", "bb_mid", "rsi", "adx", "atr", "vol_ratio"]
        if not all(col in df.columns for col in required):
            return signals

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = df.index[i]

            if pd.isna(row["adx"]) or pd.isna(row["bb_upper"]):
                continue

            # Only trade in ranging/low-trend markets
            if row["adx"] > p["adx_max"]:
                continue

            # Need volume confirmation for reversals
            if row["vol_ratio"] < p["min_volume_ratio"]:
                continue

            # Oversold bounce — price at/below lower band + RSI oversold
            if row["close"] <= row["bb_lower"] and row["rsi"] <= p["rsi_oversold"]:
                confidence = min(1.0, (p["rsi_oversold"] - row["rsi"]) / 20 * 0.4 + 0.5)
                signals.append(TradeSignal(
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    symbol=df.attrs.get("symbol", ""),
                    signal=Signal.BUY,
                    confidence=confidence,
                    entry_price=row["close"],
                    stop_loss=row["close"] - p["atr_sl_multiplier"] * row["atr"],
                    take_profit=row["bb_mid"],  # Target the mean
                    module_name=self.name,
                    reasoning=f"BB lower touch + RSI={row['rsi']:.1f} oversold, target mid={row['bb_mid']:.2f}",
                ))

            # Overbought reversal — price at/above upper band + RSI overbought
            elif row["close"] >= row["bb_upper"] and row["rsi"] >= p["rsi_overbought"]:
                confidence = min(1.0, (row["rsi"] - p["rsi_overbought"]) / 20 * 0.4 + 0.5)
                signals.append(TradeSignal(
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    symbol=df.attrs.get("symbol", ""),
                    signal=Signal.SELL,
                    confidence=confidence,
                    entry_price=row["close"],
                    stop_loss=row["close"] + p["atr_sl_multiplier"] * row["atr"],
                    take_profit=row["bb_mid"],
                    module_name=self.name,
                    reasoning=f"BB upper touch + RSI={row['rsi']:.1f} overbought, target mid={row['bb_mid']:.2f}",
                ))

        return signals
