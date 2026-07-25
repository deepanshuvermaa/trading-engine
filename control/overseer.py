"""Overseer — the orchestrator that wires everything together.

This is the "Lead" in the agent office. It:
1. Loads config and initializes all modules
2. Fetches data for each enabled market
3. Runs the inner Karpathy loop on each module
4. Runs the outer meta-loop daily
5. Generates KRA reports
6. Routes signals to paper trader in live mode
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from backtest.engine import BacktestEngine
from config.loader import Settings, load_settings
from data.ingestion.crypto import CryptoProvider
from data.ingestion.indian_equity import IndianEquityProvider
from data.ingestion.us_equity import USEquityProvider
from data.ingestion.macro import FREDProvider, GDELTProvider
from execution.paper_trader import PaperTrader
from indicators.technical import compute_all
from loops.experiment import ExperimentLog
from loops.inner_loop import InnerLoop
from loops.outer_loop import OuterLoop
from reports.daily_report import generate_daily_report, save_report
from strategy.base import StrategyModule
from strategy.modules import ALL_MODULES
from utils.logger import get_logger

log = get_logger("overseer")


class Overseer:
    """Top-level orchestrator for the trading engine."""

    def __init__(self, config_path: str | None = None):
        self.settings = load_settings(config_path)
        self.state_dir = Path("./data/storage/module_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Data providers
        crypto_cfg = self.settings.markets.get("crypto")
        self.crypto = CryptoProvider(
            crypto_cfg.exchange if crypto_cfg else "binance",
            self.settings.engine.data_dir,
        ) if crypto_cfg and crypto_cfg.enabled else None

        indian_cfg = self.settings.markets.get("indian_equity")
        self.indian_eq = IndianEquityProvider(
            self.settings.engine.data_dir
        ) if indian_cfg and indian_cfg.enabled else None

        us_cfg = self.settings.markets.get("us_equity")
        self.us_equity = USEquityProvider(
            self.settings.engine.data_dir
        ) if us_cfg and us_cfg.enabled else None

        self.fred = FREDProvider()
        self.gdelt = GDELTProvider()

        # Backtest engine (the verifier — immutable)
        self.bt_engine = BacktestEngine(
            risk_config=self.settings.risk,
            initial_capital=100_000.0,
        )

        # Experiment log (persistent state)
        self.exp_log = ExperimentLog()

        # Karpathy loops
        self.inner_loop = InnerLoop(self.settings, self.bt_engine, self.exp_log)
        self.outer_loop = OuterLoop(self.settings, self.exp_log)

        # Strategy modules
        self.modules: list[StrategyModule] = []
        self._init_modules()

        # Paper trader
        self.paper_trader = PaperTrader(self.settings.risk)

    def _init_modules(self):
        """Initialize all strategy modules, loading saved state if available."""
        for ModuleClass in ALL_MODULES:
            module = ModuleClass()
            state_path = self.state_dir / f"{module.name}.json"
            if state_path.exists():
                module.load_state(state_path)
                log.info(f"Loaded state for {module.name} (status={module.kpi.status.value})")
            else:
                log.info(f"Initialized new module: {module.name}")
            self.modules.append(module)

    async def run_backtest_cycle(self):
        """Run the inner Karpathy loop on all modules across all markets."""
        log.info("=" * 50)
        log.info("Starting backtest optimization cycle")

        window_days = self.settings.loops.inner.backtest_window_days
        end = datetime.utcnow()
        start = end - timedelta(days=window_days)

        # Fetch data for enabled markets
        datasets: list[tuple[str, any]] = []

        if self.crypto:
            market_cfg = self.settings.markets["crypto"]
            for symbol in market_cfg.symbols[:2]:  # Limit to top 2 for speed
                for tf in market_cfg.timeframes[:1]:  # Primary timeframe
                    try:
                        df = await self.crypto.get_ohlcv(symbol, tf, start, end)
                        if len(df) > 50:
                            datasets.append((symbol, df))
                    except Exception as e:
                        log.error(f"Failed to fetch {symbol}: {e}")

        if self.indian_eq:
            market_cfg = self.settings.markets["indian_equity"]
            for symbol in market_cfg.symbols[:3]:
                try:
                    df = await self.indian_eq.get_ohlcv(symbol, "1d", start, end)
                    if len(df) > 50:
                        datasets.append((symbol, df))
                except Exception as e:
                    log.error(f"Failed to fetch {symbol}: {e}")

        if self.us_equity:
            market_cfg = self.settings.markets["us_equity"]
            for symbol in market_cfg.symbols[:3]:
                try:
                    df = await self.us_equity.get_ohlcv(symbol, "1d", start, end)
                    if len(df) > 50:
                        datasets.append((symbol, df))
                except Exception as e:
                    log.error(f"Failed to fetch {symbol}: {e}")

        log.info(f"Loaded {len(datasets)} datasets")

        # Run inner loop on each module × dataset
        for module in self.modules:
            if module.kpi.status.value in ("TERMINATED", "DEACTIVATED"):
                continue

            log.info(f"Running experiments on {module.name}...")
            for symbol, data in datasets:
                self.inner_loop.run_cycle(module, data, symbol)

            # Save state after experiments
            module.save_state(self.state_dir / f"{module.name}.json")

    def run_daily_review(self) -> str:
        """Run the outer meta-loop and generate daily report."""
        log.info("Running daily review (outer loop)...")

        # Outer loop analysis
        outer_report = self.outer_loop.analyze_and_adjust(
            self.modules, self.inner_loop
        )

        # Save module states after lifecycle updates
        for module in self.modules:
            module.save_state(self.state_dir / f"{module.name}.json")

        # Generate KRA report
        report = generate_daily_report(self.modules, self.exp_log, outer_report)
        report_path = save_report(report)
        log.info(f"Daily report saved to {report_path}")

        print(report)
        return report

    async def run_full_cycle(self):
        """Run one complete cycle: fetch → optimize → review → report."""
        await self.run_backtest_cycle()
        self.run_daily_review()

    async def shutdown(self):
        """Clean up resources."""
        if self.crypto:
            await self.crypto.close()
        self.paper_trader.save_log()
        log.info("Overseer shutdown complete")
