"""
AUTONOMOUS TRADING ENGINE
==========================
Runs until target equity is reached. No human in the loop.

1. Scan all markets
2. Score and rank setups
3. Execute best trades (paper)
4. Monitor positions (check SL/TP on every scan)
5. Learn from outcomes (memory log)
6. Adjust strategy weights
7. Compound gains
8. Repeat every SCAN_INTERVAL_MINUTES
9. STOP when equity >= target

Dashboard at http://localhost:8050
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import uvicorn

from config.loader import load_settings
from dashboard.server import (
    ENGINE_STATE, app, broadcast, update_state, append_state,
)
from data.ingestion.crypto import CryptoProvider
from data.ingestion.us_equity import USEquityProvider
from data.ingestion.indian_equity import IndianEquityProvider
from indicators.technical import compute_all
from indicators.structural import (
    support_resistance_levels, market_structure_break,
    detect_order_blocks, detect_fair_value_gaps,
)
from data.models import Signal
from control.decision_audit import DecisionAudit
from db.store import get_store
from knowledge.brain import RuleBrain
from loops.attribution import TradeAttribution
from macro.intelligence import MacroIntelligence
from personas.engine import PersonaEngine
from personas.manager import PortfolioManagerAgent, RiskManagerAgent
from data.universe import UniverseDiscovery
from utils.logger import get_logger, get_recent_logs

log = get_logger("autonomous")

# ── CONFIG (env-overridable for cloud deploys; defaults = local dev) ──

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


INITIAL_CAPITAL = _env_float("INITIAL_CAPITAL", 100.0)
TARGET_EQUITY = _env_float("TARGET_EQUITY", 120.0)
SCAN_INTERVAL_MINUTES = _env_int("SCAN_INTERVAL_MINUTES", 30)  # Daily-mode rescan cadence
MAX_POSITION_PCT = 10.0
MAX_CONCURRENT = 3
COMMISSION_PCT = 0.1
SLIPPAGE_PCT = 0.05
DASHBOARD_PORT = _env_int("PORT", 8050)  # Railway injects PORT
# Memory safety valve for small cloud plans: SKIP_FINBERT=1 forces neutral
# sentiment so torch/transformers weights are never loaded (~1.5 GB saved).
SKIP_FINBERT = _env_flag("SKIP_FINBERT")
MACRO_CACHE_MINUTES = 30       # News/macro snapshot refresh interval
NEWS_SCORE_MAX = 10            # Max +/- points news sentiment can add to a setup
UNIVERSE_CACHE_MINUTES = 30    # Dynamic universe rediscovery interval
TICKER_INTERVAL_SECONDS = 60   # (legacy) inter-cycle price mark default

# ── TWO-LOOP TIMEFRAME-MATCHED ARCHITECTURE ──────────────────────────
# Scout runs the full universe scan aligned to candle closes; Sentry marks
# and protects OPEN positions on a fast fixed cadence, always.
SCAN_TIMEFRAME = os.environ.get("SCAN_TIMEFRAME", "15m").strip() or "15m"
SENTRY_INTERVAL_SECONDS = _env_int("SENTRY_INTERVAL_SECONDS", 45)
INTRADAY_LOOKBACK_DAYS = _env_int("INTRADAY_LOOKBACK_DAYS", 30)
# Trailing stop: activate once price runs TRAIL_ACTIVATION_ATR*ATR in favour,
# then ratchet the stop to TRAIL_DISTANCE_ATR*ATR behind the best price.
TRAIL_ACTIVATION_ATR = _env_float("TRAIL_ACTIVATION_ATR", 1.5)
TRAIL_DISTANCE_ATR = _env_float("TRAIL_DISTANCE_ATR", 1.0)

# Candle length in minutes for supported Scout timeframes.
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}

# ── MARKET HOURS (internal storage stays UTC; convert only to check) ──
IST = ZoneInfo("Asia/Kolkata")
US_EASTERN = ZoneInfo("America/New_York")


def market_hours_status(now: datetime | None = None) -> dict:
    """Open/closed per market. NSE 09:15-15:30 IST Mon-Fri;
    NYSE 09:30-16:00 America/New_York (DST-safe) Mon-Fri; crypto 24/7."""
    now = now or datetime.now(timezone.utc)
    ist = now.astimezone(IST)
    ny = now.astimezone(US_EASTERN)
    nse_open = ist.weekday() < 5 and (9, 15) <= (ist.hour, ist.minute) < (15, 30)
    nyse_open = ny.weekday() < 5 and (9, 30) <= (ny.hour, ny.minute) < (16, 0)
    return {
        "india": nse_open,
        "us": nyse_open,
        "crypto": True,
        "ist_time": ist.strftime("%H:%M"),
    }


class Position:
    def __init__(self, symbol, side, entry, sl, tp, size, module, opened_at,
                 reasons, atr=0.0, trail_stop=None):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry
        self.stop_loss = sl
        self.take_profit = tp
        self.size = size
        self.module = module
        self.opened_at = opened_at
        self.reasons = reasons
        self.unrealized = 0.0
        # Trailing-stop state (Sentry ratchets this as price runs in favour).
        self.atr = float(atr or 0.0)
        self.trail_stop = trail_stop  # None until activation threshold is hit

    def effective_stop(self):
        """The stop that actually protects the position — the tighter of the
        hard stop and any ratcheted trailing stop."""
        if self.trail_stop is None:
            return self.stop_loss
        if self.side == "BUY":
            return max(self.stop_loss, self.trail_stop)
        return min(self.stop_loss, self.trail_stop)

    def to_dict(self):
        return {
            "symbol": self.symbol, "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "size": round(self.size, 6), "module": self.module,
            "opened_at": self.opened_at, "unrealized": round(self.unrealized, 4),
            "atr": round(self.atr, 4),
            "trail_stop": (round(self.trail_stop, 2)
                           if self.trail_stop is not None else None),
        }


class AutonomousEngine:
    def __init__(self):
        self.settings = load_settings()
        self.crypto = CryptoProvider("binance", self.settings.engine.data_dir)
        self.us_eq = USEquityProvider(self.settings.engine.data_dir)
        self.indian_eq = IndianEquityProvider(self.settings.engine.data_dir)

        self.store = get_store()  # Postgres persistence (no-op w/o DATABASE_URL)
        self.audit = DecisionAudit()
        self.brain = RuleBrain()  # codified investor rules — every decision cites them
        self.macro = MacroIntelligence(cache_minutes=MACRO_CACHE_MINUTES)
        self.macro_snapshot: dict | None = None  # last news/sentiment snapshot
        self.universe = UniverseDiscovery(cache_minutes=UNIVERSE_CACHE_MINUTES)
        self.symbol_markets: dict[str, str] = {}  # symbol -> crypto|us|india
        self.attribution = TradeAttribution(rule_stats=self.brain.stats)
        self._verdicts: dict[str, object] = {}  # symbol -> last RuleVerdict
        # The Partners' Room — 10 deterministic investor personas + managers
        self.personas = PersonaEngine(brain=self.brain)
        self.risk_manager = RiskManagerAgent()
        self.portfolio_manager = PortfolioManagerAgent(stats=self.personas.stats)
        self._entry_votes: dict[str, list] = {}    # symbol -> persona votes at entry
        self._position_consensus: dict[str, dict] = {}  # symbol -> consensus at entry
        # Live-tunable parameters (module constants are just the defaults;
        # the dashboard's Editor's Desk mutates these through apply_config).
        self.initial_capital = INITIAL_CAPITAL
        self.target_equity = TARGET_EQUITY
        self.scan_interval_minutes = SCAN_INTERVAL_MINUTES
        self.max_position_pct = MAX_POSITION_PCT
        self.max_concurrent = MAX_CONCURRENT
        # Two-loop / timeframe-matched controls (live-tunable via Editor's Desk)
        self.scan_timeframe = SCAN_TIMEFRAME            # "15m" (candle-aligned) or "1d"
        self.sentry_interval_seconds = SENTRY_INTERVAL_SECONDS
        self.intraday_lookback_days = INTRADAY_LOOKBACK_DAYS
        self.trail_activation_atr = TRAIL_ACTIVATION_ATR
        self.trail_distance_atr = TRAIL_DISTANCE_ATR
        self.last_sentry_run: str | None = None

        self.equity = INITIAL_CAPITAL
        self.peak_equity = INITIAL_CAPITAL
        self.drawdown_pct = 0.0
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.memory: list[dict] = []
        self.cycle = 0
        self.running = True
        self.paused = False
        self.last_scan_at: str | None = None
        self._scan_now_event = asyncio.Event()

        # Agent (module) tracking
        self.agent_stats = {
            "trend_follower": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
            "mean_reverter": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
            "breakout": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
        }

    def _mem(self, msg: str, mtype: str = "info"):
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "msg": msg, "type": mtype.upper()}
        self.memory.append(entry)
        if self.store.enabled:
            self.store.fire(self.store.log_memory(entry))
        if mtype == "FAIL":
            log.warning(msg)
        elif mtype == "SUCCESS":
            log.info(msg)
        else:
            log.info(msg)

    # ── Persistence (Postgres via db/store.py; no-op without DATABASE_URL) ──

    def _engine_state_row(self) -> dict:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "initial_capital": self.initial_capital,
            "target_equity": self.target_equity,
            "cycle": self.cycle,
            "params": {
                "scan_interval_minutes": self.scan_interval_minutes,
                "max_concurrent": self.max_concurrent,
                "max_position_pct": self.max_position_pct,
                "scan_timeframe": self.scan_timeframe,
                "sentry_interval_seconds": self.sentry_interval_seconds,
                "trail_activation_atr": self.trail_activation_atr,
                "trail_distance_atr": self.trail_distance_atr,
                "paused": self.paused,
                "agent_stats": self.agent_stats,
            },
        }

    def _persist_engine_state(self):
        """Fire-and-forget snapshot of the singleton engine_state row."""
        if self.store.enabled:
            self.store.fire(self.store.save_engine_state(self._engine_state_row()))

    async def restore_state(self):
        """Connect the store and resume from Postgres where we left off."""
        await self.store.init()
        if not self.store.enabled:
            self._mem("Persistence: DATABASE_URL not set — local JSONL/file "
                      "fallback only (state resets on restart)")
            return

        st = await self.store.load_engine_state()
        if st:
            self.equity = float(st["equity"])
            self.peak_equity = float(st["peak_equity"])
            self.initial_capital = float(st["initial_capital"])
            self.target_equity = float(st["target_equity"])
            self.cycle = int(st["cycle"])
            self.drawdown_pct = (
                (self.peak_equity - self.equity) / self.peak_equity * 100
                if self.peak_equity > 0 else 0.0)
            params = st.get("params") or {}
            self.scan_interval_minutes = int(
                params.get("scan_interval_minutes", self.scan_interval_minutes))
            self.max_concurrent = int(
                params.get("max_concurrent", self.max_concurrent))
            self.max_position_pct = float(
                params.get("max_position_pct", self.max_position_pct))
            self.scan_timeframe = str(
                params.get("scan_timeframe", self.scan_timeframe))
            self.sentry_interval_seconds = int(
                params.get("sentry_interval_seconds", self.sentry_interval_seconds))
            self.trail_activation_atr = float(
                params.get("trail_activation_atr", self.trail_activation_atr))
            self.trail_distance_atr = float(
                params.get("trail_distance_atr", self.trail_distance_atr))
            saved_agents = params.get("agent_stats") or {}
            for name, stats in saved_agents.items():
                self.agent_stats[name] = stats

        # Open positions
        for p in await self.store.load_positions():
            self.positions[p["symbol"]] = Position(
                p["symbol"], p["side"], float(p["entry_price"]),
                float(p["stop_loss"]), float(p["take_profit"]),
                float(p["size"]), p.get("module") or "unknown",
                p.get("opened_at"), list(p.get("reasons") or []),
            )

        # Journal, curve, memory — the dashboard's history
        self.closed_trades = await self.store.load_trades(limit=1000)
        self.equity_curve = await self.store.load_equity_curve(limit=500)
        db_memory = await self.store.load_memory(limit=100)
        if db_memory:
            self.memory = db_memory + self.memory

        # The learning brain reads its own history from the DB
        await self.brain.stats.load_from_db()
        await self.attribution.load_regime_perf_from_db()

        if st:
            self._mem(
                f"RESUMED from Postgres: cycle {self.cycle}, "
                f"equity ${self.equity:.2f} (peak ${self.peak_equity:.2f}), "
                f"{len(self.positions)} open positions, "
                f"{len(self.closed_trades)} journal rows, "
                f"{len(self.brain.stats.stats)} rule ledgers", "SUCCESS")
        else:
            self._mem("Persistence: Postgres connected — fresh state, "
                      "first snapshot saved")
            self._persist_engine_state()

    # ── Editor's Desk controls (called from dashboard/server.py) ────

    def apply_config(self, cfg: dict) -> dict:
        """Apply live parameter changes. Returns the subset actually applied."""
        applied: dict = {}

        def _num(key, cast=float):
            v = cfg.get(key)
            if v is None or v == "":
                return None
            try:
                return cast(v)
            except (TypeError, ValueError):
                return None

        cap = _num("capital")
        if cap is not None and cap > 0 and abs(cap - self.initial_capital) > 1e-9:
            delta = cap - self.initial_capital
            self.initial_capital = cap
            self.equity += delta
            self.peak_equity = max(self.equity, self.peak_equity + delta)
            self.drawdown_pct = (
                (self.peak_equity - self.equity) / self.peak_equity * 100
                if self.peak_equity > 0 else 0.0)
            applied["capital"] = cap
            self._mem(f"EDITOR: capital adjusted by ${delta:+.2f} to "
                      f"${cap:.2f} (equity now ${self.equity:.2f})")

        tgt = _num("target")
        if tgt is not None and tgt > 0 and tgt != self.target_equity:
            self.target_equity = tgt
            applied["target"] = tgt
            self._mem(f"EDITOR: target equity set to ${tgt:.2f}")

        iv = _num("scan_interval_minutes", int)
        if iv is not None and iv != self.scan_interval_minutes:
            self.scan_interval_minutes = max(1, min(iv, 24 * 60))
            applied["scan_interval_minutes"] = self.scan_interval_minutes
            self._mem(f"EDITOR: scan interval set to "
                      f"{self.scan_interval_minutes} min (takes effect after "
                      f"the current wait)")

        mp = _num("max_positions", int)
        if mp is not None and mp != self.max_concurrent:
            self.max_concurrent = max(1, min(mp, 50))
            applied["max_positions"] = self.max_concurrent
            self._mem(f"EDITOR: max concurrent positions set to "
                      f"{self.max_concurrent}")

        mpp = _num("max_position_pct")
        if mpp is not None and mpp != self.max_position_pct:
            self.max_position_pct = max(0.1, min(mpp, 100.0))
            applied["max_position_pct"] = self.max_position_pct
            self._mem(f"EDITOR: max position size set to "
                      f"{self.max_position_pct:.1f}% of equity")

        # ── Two-loop / timeframe controls ──
        tf = cfg.get("scan_timeframe")
        if tf is not None and tf != "":
            tf = str(tf).strip().lower()
            if tf in TIMEFRAME_MINUTES or tf == "1d":
                if tf != self.scan_timeframe:
                    self.scan_timeframe = tf
                    applied["scan_timeframe"] = tf
                    self._mem(f"EDITOR: scan timeframe set to {tf} "
                              f"({'candle-aligned Scout' if tf != '1d' else 'daily cadence'})")

        si = _num("sentry_interval_seconds", int)
        if si is not None and si != self.sentry_interval_seconds:
            self.sentry_interval_seconds = max(5, min(si, 3600))
            applied["sentry_interval_seconds"] = self.sentry_interval_seconds
            self._mem(f"EDITOR: Sentry interval set to "
                      f"{self.sentry_interval_seconds}s")

        ta = _num("trail_activation_atr")
        if ta is not None and ta != self.trail_activation_atr:
            self.trail_activation_atr = max(0.0, min(ta, 20.0))
            applied["trail_activation_atr"] = self.trail_activation_atr
            self._mem(f"EDITOR: trailing-stop activation set to "
                      f"{self.trail_activation_atr:.2f}x ATR")

        td = _num("trail_distance_atr")
        if td is not None and td != self.trail_distance_atr:
            self.trail_distance_atr = max(0.1, min(td, 20.0))
            applied["trail_distance_atr"] = self.trail_distance_atr
            self._mem(f"EDITOR: trailing-stop distance set to "
                      f"{self.trail_distance_atr:.2f}x ATR")

        if applied:
            self._persist_engine_state()
        return applied

    def trigger_scan(self):
        """Wake the run loop for an immediate cycle."""
        self._mem("EDITOR: manual scan requested — running the presses now")
        self._scan_now_event.set()

    async def set_paused(self, paused: bool):
        if paused == self.paused:
            return
        self.paused = paused
        self._mem("EDITOR: presses PAUSED — no new scans until resumed"
                  if paused else "EDITOR: presses RESUMED")
        self._persist_engine_state()
        await self.push_state()

    def market_of(self, symbol: str) -> str:
        """Classify a symbol into its news-sentiment market bucket."""
        if "/" in symbol:
            return "crypto"
        return self.symbol_markets.get(symbol, "us")

    async def fetch_universe(self) -> dict[str, pd.DataFrame]:
        """Discover today's universe dynamically, then fetch OHLCV for it."""
        start = datetime.now(timezone.utc) - timedelta(days=200)
        now = datetime.now(timezone.utc)

        uni = await self.universe.discover()
        self.symbol_markets = {
            s: mkt for mkt in ("crypto", "us", "india") for s in uni.get(mkt, [])
        }
        ENGINE_STATE["universe"] = {
            "total": uni.get("total", 0),
            "counts": uni.get("counts", {}),
            "sources": uni.get("sources", {}),
            "updated_at": uni.get("updated_at"),
        }
        self._mem(
            f"Universe: {uni.get('total', 0)} securities under watch — "
            f"{uni.get('counts', {}).get('us', 0)} NYSE/NASDAQ, "
            f"{uni.get('counts', {}).get('india', 0)} NSE, "
            f"{uni.get('counts', {}).get('crypto', 0)} crypto"
        )

        # Market-hours gate: only fetch/trade markets that are open right now.
        # Crypto never sleeps; NSE and NYSE keep banker's hours.
        hours = market_hours_status()
        ENGINE_STATE["market_hours"] = hours
        fetch_markets = ["crypto"]
        if hours["india"]:
            fetch_markets.append("india")
        else:
            self._mem("NSE closed — skipping Indian equities this scan")
        if hours["us"]:
            fetch_markets.append("us")
        else:
            self._mem("NYSE closed — skipping US equities this scan")

        providers = {"crypto": self.crypto, "us": self.us_eq, "india": self.indian_eq}
        intraday = self.scan_timeframe != "1d"

        async def fetch_one(market: str, symbol: str):
            try:
                if intraday:
                    # 15m entry signals — all three markets via yfinance.
                    df = await providers[market].fetch_intraday(
                        symbol, self.scan_timeframe, self.intraday_lookback_days)
                else:
                    df = await providers[market].get_ohlcv(
                        symbol, "1d", start, now, use_cache=True)
                if df is not None and len(df) > 50:
                    return symbol, df
            except Exception as e:
                # yfinance 15m rate-limits — skip this asset this cycle, don't crash
                log.warning(f"Scout fetch skipped {symbol} ({market}, "
                            f"{self.scan_timeframe}): {e}")
            return symbol, None

        tasks = [
            fetch_one(mkt, s)
            for mkt in fetch_markets
            for s in uni.get(mkt, [])
        ]
        datasets: dict[str, pd.DataFrame] = {}
        for sym, df in await asyncio.gather(*tasks):
            if df is not None:
                datasets[sym] = df

        # Proof-of-life log: which timeframe, how many assets, a crypto sample.
        sample = next((f"{s}={len(datasets[s])} bars"
                       for s in datasets if "/" in s), "none")
        self._mem(f"Scout data: {len(datasets)} assets on {self.scan_timeframe} "
                  f"bars (crypto sample {sample})")
        return datasets

    async def refresh_macro_intel(self, force: bool = False):
        """Fetch news/macro sentiment (30-min cached inside MacroIntelligence)."""
        try:
            snap = await self.macro.get_snapshot(
                force=force, skip_finbert=SKIP_FINBERT)
            self.macro_snapshot = snap
            ENGINE_STATE["news"] = {
                "sentiments": snap.get("sentiments", {}),
                "headlines": snap.get("headlines", []),
                "all_articles": snap.get("all_articles", []),
                "updated_at": snap.get("updated_at"),
            }
            s = snap.get("sentiments", {})
            heads = snap.get("headlines", [])
            top = heads[0]["title"][:90] if heads else "no headlines"
            self._mem(
                f"News: crypto sentiment {s.get('crypto', 0):+.2f}, "
                f"india {s.get('india', 0):+.2f}, us {s.get('us', 0):+.2f}, "
                f"geo-risk {s.get('geopolitical_risk', 0):.2f}, "
                f"headline: {top}"
            )
            self.audit.log_system_event("MACRO_INTELLIGENCE", {
                "sentiments": s,
                "article_count": snap.get("article_count", 0),
                "gdelt_tone": snap.get("gdelt_tone", {}),
                "top_headlines": [
                    {"title": h["title"], "source": h["source"],
                     "sentiment": h["sentiment"]}
                    for h in heads[:5]
                ],
            })
        except Exception as e:
            self._mem(f"Macro intel unavailable this cycle: {e}", "FAIL")

    def score_asset(self, symbol: str, df: pd.DataFrame,
                    macro: dict | None = None) -> dict | None:
        if len(df) < 50:
            return None
        enriched = compute_all(df)
        last = enriched.iloc[-1]
        prev = enriched.iloc[-2] if len(enriched) > 1 else last

        for col in ["rsi", "atr", "adx"]:
            if pd.isna(last.get(col)):
                return None

        price = float(last["close"])
        r = float(last["rsi"])
        a = float(last["adx"])
        mh = float(last["macd_hist"]) if not pd.isna(last.get("macd_hist")) else 0
        pmh = float(prev["macd_hist"]) if not pd.isna(prev.get("macd_hist")) else 0
        at = float(last["atr"])
        vr = float(last["vol_ratio"]) if not pd.isna(last.get("vol_ratio")) else 1
        ef = float(last["ema_fast"]) if not pd.isna(last.get("ema_fast")) else 0
        es = float(last["ema_slow"]) if not pd.isna(last.get("ema_slow")) else 0
        pef = float(prev["ema_fast"]) if not pd.isna(prev.get("ema_fast")) else 0
        pes = float(prev["ema_slow"]) if not pd.isna(prev.get("ema_slow")) else 0
        s50 = float(last["sma_50"]) if not pd.isna(last.get("sma_50")) else 0
        s200 = float(last["sma_200"]) if not pd.isna(last.get("sma_200")) else 0
        bbu = float(last["bb_upper"]) if not pd.isna(last.get("bb_upper")) else 0
        bbl = float(last["bb_lower"]) if not pd.isna(last.get("bb_lower")) else 0

        score = 0
        reasons = []

        # Trend
        if s200 > 0 and price > s200: score += 15; reasons.append("Above SMA200")
        elif s200 > 0: score -= 15; reasons.append("Below SMA200")
        if s50 > 0 and s200 > 0 and s50 > s200: score += 10; reasons.append("Golden cross")
        elif s50 > 0 and s200 > 0: score -= 10; reasons.append("Death cross")

        # EMA cross
        if ef > es and pef <= pes: score += 20; reasons.append("FRESH bullish EMA cross")
        elif ef < es and pef >= pes: score -= 20; reasons.append("FRESH bearish EMA cross")
        elif ef > es: score += 5; reasons.append("EMA bullish")
        elif ef < es: score -= 5; reasons.append("EMA bearish")

        # RSI
        if r < 30: score += 20; reasons.append(f"RSI {r:.0f} OVERSOLD")
        elif r < 40: score += 10; reasons.append(f"RSI {r:.0f} near oversold")
        elif r > 70: score -= 20; reasons.append(f"RSI {r:.0f} OVERBOUGHT")
        elif r > 60: score -= 10; reasons.append(f"RSI {r:.0f} near overbought")

        # MACD flip
        if mh > 0 and pmh <= 0: score += 15; reasons.append("MACD turned positive")
        elif mh < 0 and pmh >= 0: score -= 15; reasons.append("MACD turned negative")

        # BB
        if bbl > 0 and price <= bbl: score += 15; reasons.append("At BB lower")
        elif bbu > 0 and price >= bbu: score -= 15; reasons.append("At BB upper")

        # Volume
        if vr > 1.5:
            score = int(score * 1.15)
            reasons.append(f"Vol {vr:.1f}x surge")

        # ADX amplify
        if a > 25: score = int(score * 1.2); reasons.append(f"ADX {a:.0f} trending")

        # MSB
        msb = market_structure_break(enriched["high"], enriched["low"])
        m = int(msb.iloc[-1]) if not pd.isna(msb.iloc[-1]) else 0
        if m == 1: score += 15; reasons.append("Bullish structure break")
        elif m == -1: score -= 15; reasons.append("Bearish structure break")

        # News/macro sentiment modifier — market-matched, capped at +/- NEWS_SCORE_MAX
        news_dir = None
        if macro:
            market = self.market_of(symbol)
            news_dir = (macro.get("sentiments") or {}).get(market)
            if news_dir is not None and news_dir != 0:
                mod = int(round(max(-1.0, min(1.0, news_dir)) * NEWS_SCORE_MAX))
                if mod:
                    score += mod
                    reasons.append(
                        f"News: {market} sentiment {news_dir:+.2f} ({mod:+d} pts)")

        pct5 = (price - float(enriched["close"].iloc[-6])) / float(enriched["close"].iloc[-6]) * 100
        pct20 = (price - float(enriched["close"].iloc[-21])) / float(enriched["close"].iloc[-21]) * 100 if len(enriched) > 21 else 0

        if abs(score) < 25:
            return None

        # SL 2x ATR, TP 4.2x ATR -> 2.1:1 reward:risk, clears the immutable
        # Minervini V10 >=2:1 floor with a small margin so rounding never vetoes.
        direction = "BUY" if score > 0 else "SELL"
        if direction == "BUY":
            sl = price - 2 * at
            tp = price + 4.2 * at
        else:
            sl = price + 2 * at
            tp = price - 4.2 * at

        rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0
        module = "trend_follower" if abs(score) > 50 else "breakout" if vr > 1.3 else "mean_reverter"

        # Extra structure for the RuleBrain (Minervini trend template, 52-week rules)
        closes = enriched["close"]
        s150 = float(closes.rolling(150).mean().iloc[-1]) if len(closes) >= 150 else None
        s200_1m = None
        if "sma_200" in enriched.columns and len(enriched) >= 222:
            v = enriched["sma_200"].iloc[-22]
            s200_1m = float(v) if not pd.isna(v) else None
        w52 = min(len(enriched), 252)
        high_52w = float(enriched["high"].iloc[-w52:].max())
        low_52w = float(enriched["low"].iloc[-w52:].min())

        result = {
            "symbol": symbol, "direction": direction, "score": score,
            "confidence": min(1.0, abs(score)/100), "current_price": round(price, 2),
            "stop_loss": round(sl, 2), "take_profit": round(tp, 2),
            "risk_reward": round(rr, 2), "atr": round(at, 2),
            "rsi": round(r, 1), "adx": round(a, 1), "volume_ratio": round(vr, 2),
            "pct_5d": round(pct5, 2), "pct_20d": round(pct20, 2),
            "reasons": reasons, "module": module,
            "sma_50": s50 or None, "sma_150": s150, "sma_200": s200 or None,
            "sma_200_1m_ago": s200_1m, "high_52w": high_52w, "low_52w": low_52w,
            "market_sentiment": news_dir,
        }

        # ── Consult the RuleBrain: every decision carries rule citations ──
        ctx = self._rule_context(result)
        verdict = self.brain.evaluate_setup(result, ctx)
        self._verdicts[symbol] = verdict
        result["rule_citations"] = verdict.citations
        result["rule_failures"] = [c.citation for c in verdict.failed]
        result["rule_score_multiplier"] = verdict.score_multiplier

        if verdict.vetoed:
            # IMMUTABLE risk rule failed — hard veto, non-negotiable
            self.audit.log_rule_citations(symbol, "VETO", verdict.citations, verdict.veto_citations)
            self.audit.log_trade_decision(
                action="REJECT", symbol=symbol, setup=result,
                portfolio_context={"equity": round(self.equity, 4),
                                   "drawdown_pct": round(self.drawdown_pct, 4),
                                   "open_positions": len(self.positions)},
                reasoning="IMMUTABLE rule veto: " + " || ".join(verdict.veto_citations),
            )
            self._mem(f"VETO {symbol}: {verdict.veto_citations[0]}", "FAIL")
            return None

        # LEARNABLE rules shade the score (weights come from knowledge/rule_stats.json)
        adjusted = int(score * verdict.score_multiplier)
        if adjusted != score:
            reasons.append(f"RuleBrain x{verdict.score_multiplier:.2f} "
                           f"({len(verdict.passed)}/{len(verdict.checks)} rules pass)")
        result["score"] = adjusted
        result["confidence"] = min(1.0, abs(adjusted) / 100)
        if abs(adjusted) < 25:
            return None

        # ── The Partners' Room: 10 personas vote, the managers synthesize ──
        # Deterministic persona consensus becomes a score modifier (±15 pts)
        # and the full vote breakdown rides with the scan result / decision.
        votes = [v.model_dump() for v in
                 self.personas.evaluate(result, ctx, verdict)]
        risk_review = self.risk_manager.review(result, ctx, verdict)
        synthesis = self.portfolio_manager.synthesize(result, votes, risk_review)
        result["persona_votes"] = votes
        result["persona_risk"] = risk_review
        result["persona_consensus"] = synthesis

        mod = synthesis.get("score_modifier", 0)
        if mod:
            final = adjusted + mod
            reasons.append(f"Partners' room {synthesis['dissent']}: "
                           f"{synthesis['action']} ({mod:+d} pts)")
            if final == 0 or (final > 0) != (adjusted > 0):
                # Consensus reversed the setup's sign — the idea dies here.
                self.audit.log_trade_decision(
                    action="REJECT", symbol=symbol, setup=result,
                    portfolio_context={"equity": round(self.equity, 4),
                                       "open_positions": len(self.positions)},
                    reasoning=f"Persona consensus reversed the signal: {synthesis['summary']}",
                )
                return None
            result["score"] = final
            result["confidence"] = min(1.0, abs(final) / 100)
            if abs(final) < 25:
                return None

        return result

    def _rule_context(self, setup: dict) -> dict:
        """Portfolio/market context handed to the RuleBrain evaluators."""
        price = setup["current_price"]
        stop_pct = abs(price - setup["stop_loss"]) / price * 100 if price else 0.0
        # Planned sizing mirrors open_position(): notional capped at max_position_pct,
        # so realized risk = notional% x stop%.
        planned_notional_pct = self.max_position_pct
        planned_risk_pct = planned_notional_pct * stop_pct / 100.0

        gross = 0.0
        if self.equity > 0:
            gross = sum(p.entry_price * p.size for p in self.positions.values()) / self.equity * 100
        closed = [t for t in self.closed_trades if t.get("pnl") is not None]
        last10 = closed[-10:]
        wins10 = sum(1 for t in last10 if (t.get("pnl") or 0) > 0)
        pos = self.positions.get(setup["symbol"])

        return {
            "equity": self.equity,
            "drawdown_pct": self.drawdown_pct,
            "total_return_pct": (self.equity - self.initial_capital) / self.initial_capital * 100,
            "gross_exposure_pct": gross,
            "planned_notional_pct": planned_notional_pct,
            "planned_risk_pct": planned_risk_pct,
            "open_positions": len(self.positions),
            "max_concurrent": self.max_concurrent,
            "last10_win_rate": wins10 / len(last10) if last10 else None,
            "last10_n": len(last10),
            "existing_position_unrealized": pos.unrealized if pos else None,
            "module_stats": self.agent_stats.get(setup["module"]),
        }

    def _update_trailing_stop(self, pos: "Position", price: float) -> bool:
        """Ratchet the trailing stop as price runs in the position's favour.
        Returns True if the trail moved this tick. Only ever tightens."""
        atr = getattr(pos, "atr", 0.0) or 0.0
        if atr <= 0 or self.trail_distance_atr <= 0:
            return False
        activation = self.trail_activation_atr * atr
        distance = self.trail_distance_atr * atr
        moved = False
        if pos.side == "BUY":
            if price - pos.entry_price >= activation:
                new_trail = price - distance
                if pos.trail_stop is None or new_trail > pos.trail_stop:
                    pos.trail_stop = round(new_trail, 4)
                    moved = True
        else:
            if pos.entry_price - price >= activation:
                new_trail = price + distance
                if pos.trail_stop is None or new_trail < pos.trail_stop:
                    pos.trail_stop = round(new_trail, 4)
                    moved = True
        return moved

    @staticmethod
    def _stop_reason(pos: "Position") -> str:
        """Whether the binding stop is the ratcheted trail or the hard stop."""
        if pos.trail_stop is None:
            return "STOP_LOSS"
        if pos.side == "BUY":
            return "TRAILING_STOP" if pos.trail_stop >= pos.stop_loss else "STOP_LOSS"
        return "TRAILING_STOP" if pos.trail_stop <= pos.stop_loss else "STOP_LOSS"

    def check_positions(self, datasets: dict[str, pd.DataFrame]):
        """Scout-side SL/TP/trailing check on the freshest candle (wick-accurate
        via high/low). Sentry protects between scans; both pop from the same
        dict so a position can never be double-closed."""
        to_close = []
        for sym, pos in self.positions.items():
            if sym not in datasets:
                continue
            df = datasets[sym]
            last = df.iloc[-1]
            high = float(last["high"])
            low = float(last["low"])
            close = float(last["close"])

            # Update unrealized
            if pos.side == "BUY":
                pos.unrealized = (close - pos.entry_price) * pos.size
            else:
                pos.unrealized = (pos.entry_price - close) * pos.size

            # Ratchet trailing stop against the bar's favourable extreme.
            self._update_trailing_stop(pos, high if pos.side == "BUY" else low)
            stop = pos.effective_stop()

            # Check stops (trailing stop is folded into the effective stop)
            if pos.side == "BUY":
                if low <= stop:
                    to_close.append((sym, stop, self._stop_reason(pos)))
                elif high >= pos.take_profit:
                    to_close.append((sym, pos.take_profit, "TAKE_PROFIT"))
            else:
                if high >= stop:
                    to_close.append((sym, stop, self._stop_reason(pos)))
                elif low <= pos.take_profit:
                    to_close.append((sym, pos.take_profit, "TAKE_PROFIT"))

        for sym, exit_price, reason in to_close:
            self._close_position(sym, exit_price, reason)

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        # Slippage + commission
        slip = exit_price * SLIPPAGE_PCT / 100
        exit_price = exit_price - slip if pos.side == "BUY" else exit_price + slip
        commission = pos.size * exit_price * COMMISSION_PCT / 100

        if pos.side == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.size - commission
        else:
            pnl = (pos.entry_price - exit_price) * pos.size - commission

        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.drawdown_pct = (self.peak_equity - self.equity) / self.peak_equity * 100

        trade = {
            "id": f"T{len(self.closed_trades)+1:04d}",
            "date": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "side": pos.side,
            "entry": pos.entry_price, "exit": round(exit_price, 2),
            "size": pos.size, "pnl": round(pnl, 4),
            "reason": reason, "module": pos.module,
        }
        self.closed_trades.append(trade)
        if self.store.enabled:
            self.store.fire(self.store.delete_position(symbol))

        # Agent stats
        stats = self.agent_stats.get(pos.module, {"wins":0,"losses":0,"pnl":0,"trades":0})
        stats["trades"] += 1
        stats["pnl"] += pnl
        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        self.agent_stats[pos.module] = stats

        # Audit — analyst-grade outcome log
        self.audit.log_trade_outcome(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=round(exit_price, 2),
            pnl=round(pnl, 4), exit_reason=reason, duration_hours=0,
            entry_reasons=pos.reasons,
            market_at_exit={"rsi": 0, "adx": 0},  # filled from latest data when available
        )

        # Attribution — decompose P&L (§9.1) and update per-rule win/loss stats
        attribution = self.attribution.on_trade_close(
            trade_id=symbol, exit_price=round(exit_price, 2), exit_reason=reason,
            pnl=round(pnl, 4), size=pos.size, entry_price=pos.entry_price,
        )
        if attribution:
            self._mem(
                f"ATTRIBUTION {symbol}: thesis {attribution.thesis_pnl:+.4f}, "
                f"execution {attribution.timing_pnl:+.4f}, regime={attribution.regime}",
                "info",
            )

        # Partners' Room bookkeeping: score each persona's entry vote against
        # the realized outcome (accuracy feeds the PM's vote weights).
        entry_votes = self._entry_votes.pop(symbol, None)
        self._position_consensus.pop(symbol, None)
        if entry_votes:
            self.personas.stats.record_trade_outcome(
                entry_votes, pos.side, pnl > 0)
            self.personas.stats.save()

        # Persist the closed trade (full detail + rule citations) and the
        # post-close equity snapshot.
        if self.store.enabled:
            db_trade = dict(trade)
            db_trade["rule_citations"] = (
                list(attribution.rule_citations) if attribution else [])
            self.store.fire(self.store.save_trade(db_trade))
            self._persist_engine_state()

        if pnl > 0:
            self._mem(f"WIN ${pnl:+.4f} on {symbol} ({reason}). {', '.join(pos.reasons[:2])}", "SUCCESS")
        else:
            self._mem(f"LOSS ${pnl:+.4f} on {symbol} ({reason}). Revisit: {', '.join(pos.reasons[:2])}", "FAIL")

    def open_position(self, setup: dict):
        if setup["symbol"] in self.positions:
            return
        if len(self.positions) >= self.max_concurrent:
            return

        price = setup["current_price"]
        sl = setup["stop_loss"]
        risk_per_unit = abs(price - sl)
        if risk_per_unit <= 0:
            return

        # Size: risk max_position_pct of equity
        risk_amount = self.equity * self.max_position_pct / 100
        size = risk_amount / risk_per_unit
        notional = size * price
        max_notional = self.equity * self.max_position_pct / 100
        if notional > max_notional:
            size = max_notional / price

        # Slippage + commission on entry
        slip = price * SLIPPAGE_PCT / 100
        entry = price + slip if setup["direction"] == "BUY" else price - slip
        commission = size * entry * COMMISSION_PCT / 100
        self.equity -= commission

        pos = Position(
            setup["symbol"], setup["direction"], round(entry, 2),
            sl, setup["take_profit"], round(size, 6),
            setup["module"], datetime.now(timezone.utc).isoformat(),
            setup["reasons"], atr=float(setup.get("atr") or 0.0),
        )
        self.positions[setup["symbol"]] = pos

        open_row = {
            "id": f"T{len(self.closed_trades)+1:04d}",
            "date": datetime.now(timezone.utc).isoformat(),
            "symbol": setup["symbol"], "side": setup["direction"],
            "entry": round(entry, 2), "exit": None,
            "size": round(size, 6), "pnl": None,
            "reason": "OPEN", "module": setup["module"],
        }
        self.closed_trades.append(open_row)

        # Persist: open position row, OPEN journal row (with rule citations),
        # and the post-commission equity snapshot.
        if self.store.enabled:
            self.store.fire(self.store.upsert_position(
                pos.to_dict() | {"reasons": pos.reasons}))
            db_row = dict(open_row)
            db_row["rule_citations"] = list(setup.get("rule_citations", []))
            # Persona vote breakdown lands in the trades.detail jsonb column.
            db_row["persona_consensus"] = setup.get("persona_consensus")
            db_row["persona_votes"] = [
                {"persona": v.get("persona"), "signal": v.get("signal"),
                 "confidence": v.get("confidence")}
                for v in setup.get("persona_votes", [])]
            self.store.fire(self.store.save_trade(db_row))
            self._persist_engine_state()

        # Remember the partners' verdict for this position: scored at close,
        # shown on the front page's PARTNERS' CONSENSUS column meanwhile.
        self._entry_votes[setup["symbol"]] = list(setup.get("persona_votes", []))
        cons = setup.get("persona_consensus")
        if cons:
            self._position_consensus[setup["symbol"]] = dict(cons) | {
                "at": datetime.now(timezone.utc).isoformat()}
            self.audit.log_system_event("PERSONA_CONSENSUS", {
                "symbol": setup["symbol"], "action": cons.get("action"),
                "consensus": cons.get("consensus"), "dissent": cons.get("dissent"),
                "vetoed": cons.get("vetoed"),
            })

        # Audit — full decision reasoning + investor-rule citations
        risk_amount = size * abs(entry - sl)
        portfolio_context = {
            "equity": round(self.equity, 4),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "open_positions": len(self.positions),
            "position_size": round(size * entry, 4),
            "risk_amount": round(risk_amount, 4),
            "risk_pct": round(risk_amount / self.equity * 100, 2),
        }
        citations = setup.get("rule_citations", [])
        self.audit.log_trade_decision(
            action="OPEN", symbol=setup["symbol"], setup=setup,
            portfolio_context=portfolio_context,
            reasoning=(
                f"Score {setup['score']} ({setup['direction']}) triggered by: "
                f"{'; '.join(setup['reasons'])}. "
                f"R:R={setup['risk_reward']:.1f}, confidence={setup['confidence']:.0%}. "
                f"Module: {setup['module']}. "
                f"Investor rules: {len(citations)} evaluated, "
                f"{len(setup.get('rule_failures', []))} failed (none IMMUTABLE)."
            ),
        )
        self.audit.log_rule_citations(setup["symbol"], "OPEN", citations,
                                      setup.get("rule_failures", []))

        # Attribution — store the causal record (§9.1) for close-time decomposition
        self.attribution.record_entry(
            trade_id=setup["symbol"], setup=setup,
            context=portfolio_context | {"equity": self.equity},
            verdict=self._verdicts.get(setup["symbol"]),
        )

        self._mem(
            f"OPEN {setup['direction']} {setup['symbol']} @ ${entry:.2f} "
            f"SL=${sl} TP=${setup['take_profit']} ({setup['module']}, score={setup['score']})",
            "info"
        )

    async def push_state(self):
        """Push current state to dashboard."""
        agent_kpis = []
        for name, stats in self.agent_stats.items():
            t = stats["trades"]
            w = stats["wins"]
            agent_kpis.append({
                "name": name,
                "status": "ACTIVE" if t > 0 else "TRIAL",
                "sharpe": 0,
                "win_rate": w/t if t > 0 else 0,
                "pnl": stats["pnl"],
                "trades": t,
                "composite": (w/t*10) if t > 0 else 0,
                "weight": 1.0 if stats["pnl"] > 0 else 0.5,
                "exp_accepted": w,
                "exp_total": t,
            })

        unrealized = sum(p.unrealized for p in self.positions.values())
        state = {
            "status": ("TARGET_HIT" if self.equity >= self.target_equity
                       else "PAUSED" if self.paused else "RUNNING"),
            "capital": self.initial_capital,
            "target": self.target_equity,
            "scan_interval_minutes": self.scan_interval_minutes,
            "max_positions": self.max_concurrent,
            "max_position_pct": self.max_position_pct,
            "scan_timeframe": self.scan_timeframe,
            "sentry_interval_seconds": self.sentry_interval_seconds,
            "trail_activation_atr": self.trail_activation_atr,
            "trail_distance_atr": self.trail_distance_atr,
            "last_sentry_run": self.last_sentry_run,
            "paused": self.paused,
            "equity": round(self.equity, 4),
            "equity_mark": round(self.equity + unrealized, 4),
            "unrealized_pnl": round(unrealized, 4),
            "peak_equity": round(self.peak_equity, 4),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "total_pnl": round(self.equity - self.initial_capital, 4),
            "total_return_pct": round((self.equity - self.initial_capital) / self.initial_capital * 100, 2),
            "target_achieved": self.equity >= self.target_equity,
            "open_positions": [p.to_dict() for p in self.positions.values()],
            "trade_journal": self.closed_trades[-100:],
            "equity_curve": self.equity_curve[-500:],
            "scan_results": ENGINE_STATE.get("scan_results", []),
            "news": ENGINE_STATE.get(
                "news", {"sentiments": {}, "headlines": [], "updated_at": None}),
            "universe": ENGINE_STATE.get(
                "universe", {"total": 0, "counts": {}, "sources": {},
                             "updated_at": None}),
            "agent_kpis": agent_kpis,
            "rule_stats": self.brain.stats.snapshot(),
            "personas": ENGINE_STATE.get(
                "personas", {"votes_by_symbol": {}, "position_consensus": {},
                             "persona_records": self.personas.stats.records(),
                             "last_updated": None}),
            "memory_log": self.memory[-100:],
            "logs": get_recent_logs(50),
            "last_heartbeat": ENGINE_STATE.get("last_heartbeat"),
            "cycle_count": self.cycle,
            "last_scan": self.last_scan_at,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "market_hours": market_hours_status(),
            "errors": [],
        }
        ENGINE_STATE.update(state)
        await broadcast(state)

    async def run_cycle(self):
        """One full scan-trade-monitor cycle."""
        self.cycle += 1
        self._mem(f"--- Cycle {self.cycle} | Equity: ${self.equity:.2f} | Target: ${self.target_equity:.2f} ---")

        # 1. Fetch market data + news/macro intelligence
        _hrs = market_hours_status()
        _open = [m for m in ("crypto", "india", "us") if _hrs.get(m)]
        if self.scan_timeframe != "1d":
            self._mem(f"Scout: {self.scan_timeframe} candle closed — scanning "
                      f"{len(_open)} open market(s): {', '.join(_open)}")
        else:
            self._mem(f"Scanning all markets... ({len(_open)} open: "
                      f"{', '.join(_open)})")
        await self.refresh_macro_intel()  # 30-min cached
        datasets = await self.fetch_universe()
        self._mem(f"Loaded {len(datasets)} assets")

        # 2. Check existing positions
        if self.positions:
            self.check_positions(datasets)

        # 3. Scan for new setups (news sentiment shades the score, +/-10 pts)
        setups = []
        scan_results = []
        for sym, df in datasets.items():
            result = self.score_asset(sym, df, macro=self.macro_snapshot)
            if result:
                scan_results.append(result)
                if abs(result["score"]) >= 25 and sym not in self.positions:
                    setups.append(result)

        # 3b. Company-level headlines (yfinance, flaky — best effort, cached).
        # Cap at the 15 highest-|score| results to bound cycle time.
        top_by_score = sorted(
            scan_results, key=lambda r: abs(r.get("score", 0)), reverse=True)[:15]
        for sr in top_by_score:
            try:
                news = await self.macro.company_news(
                    sr["symbol"], self.market_of(sr["symbol"]),
                    skip_finbert=SKIP_FINBERT)
            except Exception:
                news = None
            if news and news.get("count"):
                sr["news_sentiment"] = news["direction"]
                sr["news_headlines"] = news["headlines"]
                sr["reasons"].append(
                    f"Company news {news['direction']:+.2f} "
                    f"({news['count']} headlines)")

        ENGINE_STATE["scan_results"] = scan_results
        self._mem(f"Found {len(setups)} actionable setups")

        # 3c. The Partners' Room — publish the vote breakdown for the top
        # setups plus each persona's running track record.
        top_voted = sorted(
            scan_results, key=lambda r: abs(r.get("score", 0)), reverse=True)[:10]
        ENGINE_STATE["personas"] = {
            "votes_by_symbol": {
                r["symbol"]: {
                    "direction": r.get("direction"),
                    "score": r.get("score"),
                    "current_price": r.get("current_price"),
                    "votes": r.get("persona_votes", []),
                    "risk": r.get("persona_risk"),
                    "consensus": r.get("persona_consensus"),
                }
                for r in top_voted if r.get("persona_votes")
            },
            "position_consensus": dict(self._position_consensus),
            "persona_records": self.personas.stats.records(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        # Log all scans to audit
        for sr in scan_results:
            self.audit.log_scan(sr["symbol"], sr)

        # 4. Rank and execute
        if setups and len(self.positions) < self.max_concurrent:
            ranked = sorted(setups, key=lambda x: abs(x["score"]) * x["confidence"] * min(x["risk_reward"], 3), reverse=True)
            slots = self.max_concurrent - len(self.positions)
            for setup in ranked[:slots]:
                self.open_position(setup)
            # Log skipped setups
            for setup in ranked[slots:]:
                self.audit.log_trade_decision(
                    action="SKIP", symbol=setup["symbol"], setup=setup,
                    portfolio_context={"equity": self.equity, "open_positions": len(self.positions)},
                    reasoning=f"Ranked lower than selected trades. Score={setup['score']}, max slots filled.",
                )

        # 5. Record equity (in-memory + Postgres when configured)
        eq_point = {
            "date": datetime.now(timezone.utc).isoformat(),
            "equity": round(self.equity, 4),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "positions": len(self.positions),
        }
        self.equity_curve.append(eq_point)
        if self.store.enabled:
            self.store.fire(self.store.log_equity_point(eq_point))
            self._persist_engine_state()

        # 6. Push to dashboard (last_scan anchors the client-side countdown)
        self.last_scan_at = datetime.now(timezone.utc).isoformat()
        await self.push_state()

        # 7. Check target
        if self.equity >= self.target_equity:
            self._mem(f"TARGET HIT! Equity ${self.equity:.2f} >= ${self.target_equity:.2f}", "SUCCESS")
            self.running = False

    # ── Inter-cycle price ticker ────────────────────────────────────

    def _yf_ticker_symbol(self, symbol: str) -> str:
        """Map an engine symbol to its yfinance ticker."""
        market = self.market_of(symbol)
        if market == "crypto":
            return f"{symbol.split('/')[0]}-USD"   # BTC/USDT -> BTC-USD
        if market == "india":
            return f"{symbol}.NS"                  # RELIANCE -> RELIANCE.NS
        return symbol

    def _fetch_last_prices_sync(self, symbols: list[str]) -> dict[str, float]:
        """Blocking helper: latest price per symbol via yfinance fast_info,
        falling back to the last daily close. Best effort — skips failures."""
        import yfinance as yf

        out: dict[str, float] = {}
        for sym in symbols:
            try:
                t = yf.Ticker(self._yf_ticker_symbol(sym))
                px = None
                try:
                    px = t.fast_info["last_price"]
                except Exception:
                    px = None
                if not px:
                    hist = t.history(period="1d")
                    if len(hist):
                        px = float(hist["Close"].iloc[-1])
                if px and px > 0:
                    out[sym] = float(px)
            except Exception:
                continue
        return out

    def _seconds_to_next_candle(self) -> float:
        """Seconds until the next Scout run. Candle-aligned in intraday mode
        (:00/:15/:30/:45 for 15m, off the UTC epoch grid); the old fixed
        cadence in daily mode. A small buffer lets the candle finish closing."""
        if self.scan_timeframe == "1d":
            return max(1, self.scan_interval_minutes) * 60.0
        minutes = TIMEFRAME_MINUTES.get(self.scan_timeframe, 15)
        period = minutes * 60
        epoch = datetime.now(timezone.utc).timestamp()
        nxt = (int(epoch // period) + 1) * period
        return max(1.0, (nxt - epoch) + 2.0)

    def _sentry_manage_position(self, sym: str, pos: "Position", px: float):
        """Mark one position to `px`, ratchet its trailing stop, and close it
        if the effective stop or take-profit is breached."""
        if pos.side == "BUY":
            pos.unrealized = (px - pos.entry_price) * pos.size
        else:
            pos.unrealized = (pos.entry_price - px) * pos.size

        moved = self._update_trailing_stop(pos, px)
        stop = pos.effective_stop()

        hit = None
        if pos.side == "BUY":
            if px <= stop:
                hit = (stop, self._stop_reason(pos))
            elif px >= pos.take_profit:
                hit = (pos.take_profit, "TAKE_PROFIT")
        else:
            if px >= stop:
                hit = (stop, self._stop_reason(pos))
            elif px <= pos.take_profit:
                hit = (pos.take_profit, "TAKE_PROFIT")

        if hit:
            log.info(f"SENTRY: {sym} hit {hit[1]} @ {px:.4f} — closing")
            self._close_position(sym, hit[0], hit[1])
        else:
            if moved:
                trail_txt = f", trail moved to {pos.trail_stop}"
            elif pos.trail_stop is not None:
                trail_txt = f", trail at {pos.trail_stop}"
            else:
                trail_txt = ""
            log.info(f"SENTRY: {sym} marked @ {px:.4f}, "
                     f"unrealized {pos.unrealized:+.4f}{trail_txt}")

    async def _sentry_tick(self):
        """One Sentry pass over OPEN positions. Processes closes here so a
        concurrent Scout scan never double-touches a position (both pop from
        self.positions; whoever is second finds it already gone)."""
        self.last_sentry_run = datetime.now(timezone.utc).isoformat()
        ENGINE_STATE["last_sentry_run"] = self.last_sentry_run
        # Heartbeat: proof the Sentry loop is alive, whatever else happens
        ENGINE_STATE["last_heartbeat"] = self.last_sentry_run
        # Only mark positions whose market is open — closed markets don't move.
        hours = market_hours_status()
        symbols = [s for s in self.positions
                   if hours.get(self.market_of(s), True)]
        if symbols:
            prices = await asyncio.to_thread(
                self._fetch_last_prices_sync, symbols)
            for sym in list(prices.keys()):
                pos = self.positions.get(sym)
                if not pos:
                    continue  # closed elsewhere in the meantime
                self._sentry_manage_position(sym, pos, prices[sym])
        # Push every tick so the front page stays live between scans.
        await self.push_state()

    async def sentry_loop(self):
        """SENTRY — capital protection. Every sentry_interval_seconds, mark
        OPEN positions to the latest price, ratchet trailing stops, and close
        anything that hit SL/TP/trail. Runs regardless of scan cadence and even
        while entry-scanning (the Scout) is paused."""
        while self.running:
            await asyncio.sleep(max(5, self.sentry_interval_seconds))
            if not self.running:
                break
            try:
                await self._sentry_tick()
            except Exception as e:
                log.warning(f"Sentry loop error: {e}")

    # Backwards-compatible alias for start_engine()'s ticker task.
    async def price_ticker(self):
        await self.sentry_loop()

    async def run(self):
        """SCOUT — the full universe scan → personas → rules → open positions.
        In intraday mode each pass is aligned to the next candle close
        (:00/:15/:30/:45 for 15m); in daily mode it keeps the old fixed cadence.
        Capital protection runs concurrently in sentry_loop()."""
        ENGINE_STATE["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        self._mem(f"Engine started. Capital=${self.initial_capital} Target=${self.target_equity}")
        if self.scan_timeframe == "1d":
            self._mem(f"Scout: daily mode, rescan every {self.scan_interval_minutes} min "
                      f"| Sentry: every {self.sentry_interval_seconds}s "
                      f"| Max positions: {self.max_concurrent}")
        else:
            self._mem(f"Scout: {self.scan_timeframe} candle-aligned scans "
                      f"| Sentry: every {self.sentry_interval_seconds}s "
                      f"| Max positions: {self.max_concurrent}")

        while self.running:
            manual = self._scan_now_event.is_set()
            self._scan_now_event.clear()

            if self.paused and not manual:
                # Presses paused — Scout idles; Sentry keeps protecting positions
                try:
                    await asyncio.wait_for(self._scan_now_event.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass
                continue

            if manual:
                self._mem("Scout: manual scan requested — running a pass now")

            try:
                await self.run_cycle()
            except Exception as e:
                self._mem(f"Cycle error: {e}", "FAIL")
                log.error(traceback.format_exc())
                self.last_scan_at = datetime.now(timezone.utc).isoformat()

            if not self.running:
                break

            wait_s = self._seconds_to_next_candle()
            if self.scan_timeframe == "1d":
                self._mem(f"Next scan in {self.scan_interval_minutes} min...")
            else:
                self._mem(f"Scout: next {self.scan_timeframe} candle close in "
                          f"~{int(wait_s)}s")
            try:
                # Sleep until the candle boundary, but wake instantly on scan-now
                await asyncio.wait_for(
                    self._scan_now_event.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                pass

        # Final state push + last durable snapshot
        await self.push_state()
        self._mem("Engine stopped.")
        if self.store.enabled:
            await self.store.save_engine_state(self._engine_state_row())
            await self.store.close()
        await self.crypto.close()


async def start_engine():
    """Run dashboard server + autonomous engine + price ticker concurrently."""
    engine = AutonomousEngine()

    # Expose the live engine to the FastAPI control endpoints
    import dashboard.server as dashboard_server
    dashboard_server.ENGINE = engine

    # Start uvicorn in background
    config = uvicorn.Config(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning")
    server = uvicorn.Server(config)

    async def run_server():
        await server.serve()

    async def run_engine():
        await asyncio.sleep(2)  # Let server start first
        # Resume from Postgres (no-op without DATABASE_URL)
        try:
            await engine.restore_state()
        except Exception as e:
            log.error(f"State restore failed — starting fresh: {e}")
        await engine.run()

    async def run_sentry():
        await asyncio.sleep(5)  # Let the first Scout cycle get underway
        await engine.sentry_loop()

    log.info(f"Dashboard: http://localhost:{DASHBOARD_PORT}")
    log.info("Starting autonomous engine (Scout + Sentry loops)...")

    await asyncio.gather(
        run_server(),
        run_engine(),   # Scout loop
        run_sentry(),   # Sentry loop
    )


if __name__ == "__main__":
    asyncio.run(start_engine())
