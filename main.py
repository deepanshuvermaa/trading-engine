"""Bitcoin Analyser — Multi-Asset Macro Trading Engine

Entry point. Run with: python main.py [command]

Commands:
    optimize    — Run one Karpathy optimization cycle (inner loop on all modules)
    review      — Run daily review (outer loop + KRA report)
    full        — Run full cycle (optimize + review)
    status      — Print current module status
    backtest    — Run backtest on a specific symbol
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from control.overseer import Overseer
from utils.logger import get_logger

log = get_logger("main")


async def cmd_optimize(overseer: Overseer):
    log.info("Running optimization cycle...")
    await overseer.run_backtest_cycle()
    log.info("Optimization complete.")


async def cmd_review(overseer: Overseer):
    log.info("Running daily review...")
    overseer.run_daily_review()


async def cmd_full(overseer: Overseer):
    log.info("Running full cycle (optimize + review)...")
    await overseer.run_full_cycle()


async def cmd_status(overseer: Overseer):
    from reports.daily_report import generate_daily_report
    report = generate_daily_report(overseer.modules, overseer.exp_log)
    print(report)


async def cmd_backtest(overseer: Overseer, symbol: str, timeframe: str = "1d"):
    from datetime import datetime, timedelta
    from indicators.technical import compute_all

    log.info(f"Backtesting all modules on {symbol} {timeframe}...")
    end = datetime.utcnow()
    start = end - timedelta(days=730)

    # Detect market from symbol format
    indian_symbols = set()
    if overseer.settings.markets.get("indian_equity"):
        indian_symbols = set(overseer.settings.markets["indian_equity"].symbols)

    if "/" in symbol:
        if not overseer.crypto:
            log.error("Crypto market not enabled")
            return
        df = await overseer.crypto.get_ohlcv(symbol, timeframe, start, end)
    elif symbol.endswith(".NS") or symbol in indian_symbols:
        if not overseer.indian_eq:
            log.error("Indian equity market not enabled")
            return
        df = await overseer.indian_eq.get_ohlcv(symbol, timeframe, start, end)
    else:
        if not overseer.us_equity:
            log.error("US equity market not enabled")
            return
        df = await overseer.us_equity.get_ohlcv(symbol, timeframe, start, end)

    if len(df) < 50:
        log.error(f"Insufficient data: only {len(df)} bars")
        return

    log.info(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    for module in overseer.modules:
        result = overseer.bt_engine.run(module, df, symbol)
        print(f"\n{'-'*50}")
        print(f"Module: {module.name}")
        print(f"  Sharpe:        {result.sharpe_ratio:.4f}")
        print(f"  Sortino:       {result.sortino_ratio:.4f}")
        print(f"  Win Rate:      {result.win_rate*100:.1f}%")
        print(f"  Max Drawdown:  -{abs(result.max_drawdown_pct):.2f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Total Trades:  {result.total_trades}")
        print(f"  Total Return:  {result.total_return_pct:.2f}%")
        print(f"  Composite:     {result.composite_score:.1f}/10")


async def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "status"

    overseer = Overseer()

    try:
        if command == "optimize":
            await cmd_optimize(overseer)
        elif command == "review":
            await cmd_review(overseer)
        elif command == "full":
            await cmd_full(overseer)
        elif command == "status":
            await cmd_status(overseer)
        elif command == "backtest":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT"
            tf = sys.argv[3] if len(sys.argv) > 3 else "1d"
            await cmd_backtest(overseer, symbol, tf)
        else:
            print(__doc__)
    finally:
        await overseer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
