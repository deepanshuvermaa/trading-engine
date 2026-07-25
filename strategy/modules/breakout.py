"""Breakout strategy — structure breaks + volume surge + momentum confirmation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data.models import Signal, TradeSignal
from strategy.base import StrategyModule


class BreakoutDetector(StrategyModule):
    name = "breakout"
    description = "Market structure break detection with volume surge and momentum confirmation"

    def default_params(self) -> dict[str, Any]:
        return {
            "lookback": 20,
            "volume_surge_threshold": 1.5,
            "atr_sl_multiplier": 2.5,
            "atr_tp_multiplier": 4.0,
            "rsi_min": 40,  # Don't buy exhausted breakouts
            "rsi_max": 60,  # Don't sell exhausted breakdowns
            "macd_confirm": True,
            "min_consolidation_bars": 5,
        }

    def get_param_ranges(self) -> dict[str, tuple[float, float, float]]:
        return {
            "lookback": (10, 40, 5),
            "volume_surge_threshold": (1.2, 3.0, 0.3),
            "atr_sl_multiplier": (1.5, 4.0, 0.5),
            "atr_tp_multiplier": (2.0, 6.0, 0.5),
            "rsi_min": (30, 50, 5),
            "rsi_max": (50, 70, 5),
            "min_consolidation_bars": (3, 10, 1),
        }

    def generate_signals(self, df: pd.DataFrame) -> list[TradeSignal]:
        p = self.params
        signals = []
        lookback = int(p["lookback"])

        required = ["close", "high", "low", "volume", "atr", "rsi", "vol_ratio", "macd_hist"]
        if not all(col in df.columns for col in required):
            return signals

        for i in range(lookback + 1, len(df)):
            row = df.iloc[i]
            ts = df.index[i]

            if pd.isna(row["atr"]) or pd.isna(row["rsi"]):
                continue

            window = df.iloc[i - lookback:i]
            resistance = window["high"].max()
            support = window["low"].min()
            price_range = resistance - support

            if price_range <= 0:
                continue

            # Check consolidation — range should be tightening
            recent_range = df.iloc[i - int(p["min_consolidation_bars"]):i]
            consolidation_ratio = (recent_range["high"].max() - recent_range["low"].min()) / price_range
            is_consolidating = consolidation_ratio < 0.6

            # Volume surge
            has_volume = row["vol_ratio"] >= p["volume_surge_threshold"]

            # Bullish breakout — close above resistance
            if (row["close"] > resistance and
                    has_volume and
                    row["rsi"] > p["rsi_min"] and
                    row["rsi"] < 80):

                macd_ok = row["macd_hist"] > 0 if p["macd_confirm"] else True
                if macd_ok:
                    confidence = 0.5
                    if is_consolidating:
                        confidence += 0.15
                    if row["vol_ratio"] > p["volume_surge_threshold"] * 1.5:
                        confidence += 0.15
                    confidence = min(1.0, confidence)

                    signals.append(TradeSignal(
                        timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        symbol=df.attrs.get("symbol", ""),
                        signal=Signal.BUY,
                        confidence=confidence,
                        entry_price=row["close"],
                        stop_loss=row["close"] - p["atr_sl_multiplier"] * row["atr"],
                        take_profit=row["close"] + p["atr_tp_multiplier"] * row["atr"],
                        module_name=self.name,
                        reasoning=f"Breakout above {resistance:.2f}, vol={row['vol_ratio']:.1f}x, consolidation={consolidation_ratio:.2f}",
                    ))

            # Bearish breakdown — close below support
            elif (row["close"] < support and
                  has_volume and
                  row["rsi"] < p["rsi_max"] and
                  row["rsi"] > 20):

                macd_ok = row["macd_hist"] < 0 if p["macd_confirm"] else True
                if macd_ok:
                    confidence = 0.5
                    if is_consolidating:
                        confidence += 0.15
                    if row["vol_ratio"] > p["volume_surge_threshold"] * 1.5:
                        confidence += 0.15
                    confidence = min(1.0, confidence)

                    signals.append(TradeSignal(
                        timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        symbol=df.attrs.get("symbol", ""),
                        signal=Signal.SELL,
                        confidence=confidence,
                        entry_price=row["close"],
                        stop_loss=row["close"] + p["atr_sl_multiplier"] * row["atr"],
                        take_profit=row["close"] - p["atr_tp_multiplier"] * row["atr"],
                        module_name=self.name,
                        reasoning=f"Breakdown below {support:.2f}, vol={row['vol_ratio']:.1f}x, consolidation={consolidation_ratio:.2f}",
                    ))

        return signals
