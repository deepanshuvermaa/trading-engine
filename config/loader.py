"""Configuration loader — reads YAML, resolves env vars, validates."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class RiskConfig(BaseModel):
    max_drawdown_pct: float = 2.0
    max_position_pct: float = 5.0
    max_portfolio_exposure_pct: float = 30.0
    scale_down_factor: float = 0.5
    recovery_streak_required: int = 3


class ModuleLifecycleConfig(BaseModel):
    trial_period_days: int = 30
    trial_min_sharpe: float = 0.5
    trial_max_drawdown_pct: float = 3.0
    active_min_sharpe: float = 0.3
    probation_days: int = 7
    max_trial_failures: int = 3


class MarketConfig(BaseModel):
    enabled: bool = True
    exchange: str = ""
    symbols: list[str] = []
    index: list[str] = []
    timeframes: list[str] = ["1d"]
    lookback_days: int = 365


class InnerLoopConfig(BaseModel):
    experiments_per_cycle: int = 10
    backtest_window_days: int = 730
    min_sharpe_improvement: float = 0.05


class OuterLoopConfig(BaseModel):
    run_interval_hours: int = 24
    stagnation_threshold_days: int = 30
    experiment_success_rate_min: float = 0.05


class LoopsConfig(BaseModel):
    inner: InnerLoopConfig = InnerLoopConfig()
    outer: OuterLoopConfig = OuterLoopConfig()


class MacroConfig(BaseModel):
    fred_api_key: str = ""
    gdelt_enabled: bool = True
    rss_feeds_update_interval_minutes: int = 60
    sentiment_model: str = "ProsusAI/finbert"


class EngineConfig(BaseModel):
    name: str = "macro-trading-engine"
    mode: str = "paper"
    log_level: str = "INFO"
    data_dir: str = "./data/storage"
    cache_dir: str = "./data/cache"


class Settings(BaseModel):
    engine: EngineConfig = EngineConfig()
    markets: dict[str, MarketConfig] = {}
    risk: RiskConfig = RiskConfig()
    modules: ModuleLifecycleConfig = ModuleLifecycleConfig()
    loops: LoopsConfig = LoopsConfig()
    macro: MacroConfig = MacroConfig()


def _resolve_env_vars(obj: Any) -> Any:
    """Replace ${VAR} patterns with environment variable values."""
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        match = pattern.match(obj)
        if match:
            return os.environ.get(match.group(1), "")
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        path = Path(__file__).parent / "settings.yaml"
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _resolve_env_vars(raw)
    markets_raw = raw.pop("markets", {})
    markets = {k: MarketConfig(**v) for k, v in markets_raw.items()}
    raw.pop("api_keys", None)
    return Settings(markets=markets, **raw)
