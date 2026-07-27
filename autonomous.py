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
from data.fx import to_usd, usd_rate, rate_source
from data.holidays import (
    is_nse_holiday, is_nyse_holiday, holiday_name_nse, holiday_name_nyse,
)
from data.costs import (
    order_cost_usd, round_trip_cost_usd, round_trip_cost_pct, cost_breakdown,
)
from indicators.technical import compute_all
from indicators.structural import (
    support_resistance_levels, market_structure_break,
    detect_order_blocks, detect_fair_value_gaps,
)
from data.models import Signal
from control.decision_audit import DecisionAudit
from db.store import get_store
from knowledge.brain import RuleBrain
from knowledge.anchor_guard import assert_anchors_untouched, AnchorTamperError
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
# Legacy flat-cost knobs (kept for backward compat / crypto defaults live in
# data/costs.py now). Actual P&L + the entry filter use the PER-MARKET model.
COMMISSION_PCT = _env_float("COMMISSION_PCT", 0.1)   # per side, % of notional
SLIPPAGE_PCT = _env_float("SLIPPAGE_PCT", 0.05)      # per fill, % of price
# A setup's target must clear the (per-market) round-trip cost drag by this
# multiple or the trade is skipped — no point entering when fees eat the edge.
COST_EDGE_MULTIPLE = _env_float("COST_EDGE_MULTIPLE", 5.0)
# Anti-churn: after closing a symbol, a cohort may not re-open it for this many
# minutes. Stops the 15m loop from repeatedly round-tripping the same name and
# bleeding fees. Env-overridable.
REENTRY_COOLDOWN_MIN = _env_float("REENTRY_COOLDOWN_MIN", 120.0)

# The self-improvement substrate (frozen anchors, experiments, state, lessons).
LOOP_DIR = Path(__file__).parent / ".loop"
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

# ── COHORT MODE ──────────────────────────────────────────────────────
# Several isolated paper portfolios in ONE process, all seeing the SAME data
# at the SAME instant (one fetch, one FinBERT, one shared learning brain), so
# the operator can see WHERE the edge lives: all-in crypto vs all-in US vs
# all-in NSE vs distributed. Internal market codes are crypto|us|india (see
# AutonomousEngine.market_of); cohort specs accept friendly aliases.
ALL_MARKETS = {"crypto", "us", "india"}
_MARKET_ALIASES = {
    "crypto": "crypto", "cryptocurrency": "crypto", "coin": "crypto",
    "us": "us", "us_equity": "us", "us_equities": "us", "usequity": "us",
    "nasdaq": "us", "nyse": "us", "america": "us",
    "india": "india", "indian_equity": "india", "indian_equities": "india",
    "nse": "india", "bse": "india",
    "all": None, "*": None, "distributed": None,
}


def _canon_markets(spec) -> set[str] | None:
    """Map a cohort market spec (str/list) to a set of internal codes, or None
    for 'all markets'. Unknown tokens are dropped; empty -> all markets."""
    if spec is None:
        return None
    if isinstance(spec, str):
        tokens = [t for t in spec.replace("+", ",").replace("|", ",").split(",")]
    else:
        tokens = list(spec)
    out: set[str] = set()
    for t in tokens:
        key = str(t).strip().lower()
        if not key:
            continue
        if key not in _MARKET_ALIASES:
            continue
        mapped = _MARKET_ALIASES[key]
        if mapped is None:
            return None  # an explicit 'all' widens the cohort to every market
        out.add(mapped)
    return out or None


# The four standard scenarios (each seeded at INITIAL_CAPITAL). DISTRIBUTED
# reproduces today's single-portfolio behaviour exactly (all markets allowed).
_DEFAULT_COHORTS = [
    ("CRYPTO-ONLY", {"crypto"}),
    ("US-ONLY", {"us"}),
    ("NSE-ONLY", {"india"}),
    ("DISTRIBUTED", None),
]


def parse_cohorts(raw: str | None) -> list[tuple[str, set[str] | None]]:
    """Resolve the COHORTS env var into [(name, market_filter), ...].

    - unset/empty            -> a single DISTRIBUTED portfolio (all markets):
                                exactly today's behaviour, nothing breaks.
    - 'default'/'4'/'all'/'standard'/'cohorts' -> the four standard cohorts.
    - JSON: [{"name": "X", "markets": ["crypto"]}, ...]
    - simple spec: 'CRYPTO-ONLY:crypto;US-ONLY:us_equity;DISTRIBUTED:all'
    """
    raw = (raw or "").strip()
    if not raw:
        return [("DISTRIBUTED", None)]
    if raw.lower() in ("default", "standard", "4", "four", "all", "cohorts", "on"):
        return [(name, mf) for name, mf in _DEFAULT_COHORTS]

    # JSON form
    if raw[0] in "[{":
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            cohorts: list[tuple[str, set[str] | None]] = []
            for item in data:
                name = str(item.get("name") or item.get("id") or "COHORT").strip()
                markets = item.get("markets", item.get("market"))
                cohorts.append((name, _canon_markets(markets)))
            if cohorts:
                return cohorts
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            log.warning(f"COHORTS: could not parse JSON ({e}); "
                        f"falling back to the four standard cohorts")
            return [(name, mf) for name, mf in _DEFAULT_COHORTS]

    # simple 'NAME:markets;NAME:markets' spec
    cohorts = []
    for chunk in raw.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, markets = chunk.split(":", 1)
        else:
            name, markets = chunk, "all"
        cohorts.append((name.strip() or "COHORT", _canon_markets(markets)))
    return cohorts or [("DISTRIBUTED", None)]


def market_hours_status(now: datetime | None = None) -> dict:
    """Open/closed per market. NSE 09:15-15:30 IST Mon-Fri;
    NYSE 09:30-16:00 America/New_York (DST-safe) Mon-Fri; crypto 24/7."""
    now = now or datetime.now(timezone.utc)
    ist = now.astimezone(IST)
    ny = now.astimezone(US_EASTERN)
    india_holiday = is_nse_holiday(ist.date())
    us_holiday = is_nyse_holiday(ny.date())
    nse_open = (ist.weekday() < 5
                and (9, 15) <= (ist.hour, ist.minute) < (15, 30)
                and not india_holiday)
    nyse_open = (ny.weekday() < 5
                 and (9, 30) <= (ny.hour, ny.minute) < (16, 0)
                 and not us_holiday)
    return {
        "india": nse_open,
        "us": nyse_open,
        "crypto": True,
        "ist_time": ist.strftime("%H:%M"),
        "india_holiday": india_holiday,
        "india_holiday_name": holiday_name_nse(ist.date()),
        "us_holiday": us_holiday,
        "us_holiday_name": holiday_name_nyse(ny.date()),
    }


class Position:
    def __init__(self, symbol, side, entry, sl, tp, size, module, opened_at,
                 reasons, atr=0.0, trail_stop=None, currency="USD", fx_rate=1.0,
                 market="us"):
        self.symbol = symbol
        self.side = side
        # Market ("crypto"/"us"/"india") is captured ONCE at open time and
        # never re-derived later — see open_position()/_close_position() for
        # why re-deriving from the shared, per-cycle-mutable symbol_markets
        # dict is unsafe (a concurrent scan can rebuild it mid-trade).
        self.market = market
        # Prices are stored in the asset's NATIVE currency (INR for NSE, USD
        # elsewhere) so all stop/target/trailing comparisons stay native-to-
        # native and unchanged. Money flows (unrealized/realized P&L, notional,
        # commission) are converted to true USD via fx_rate at booking time.
        self.entry_price = entry
        self.stop_loss = sl
        self.take_profit = tp
        self.size = size
        self.module = module
        self.opened_at = opened_at
        self.reasons = reasons
        self.unrealized = 0.0  # always TRUE USD
        # Trailing-stop state (Sentry ratchets this as price runs in favour).
        self.atr = float(atr or 0.0)
        self.trail_stop = trail_stop  # None until activation threshold is hit
        # Currency of the native price + native-units-per-USD at entry. USD
        # assets carry ("USD", 1.0) so their behaviour is byte-for-byte unchanged.
        self.currency = currency or "USD"
        self.fx_rate = float(fx_rate or 1.0)
        # Estimated per-market round-trip cost for this position (TRUE USD / %).
        self.est_round_trip_cost_usd = 0.0
        self.est_round_trip_cost_pct = 0.0

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
            "currency": self.currency, "fx_rate": round(self.fx_rate, 4),
            "est_round_trip_cost_usd": round(self.est_round_trip_cost_usd, 4),
            "est_round_trip_cost_pct": round(self.est_round_trip_cost_pct, 3),
        }


class Portfolio:
    """One isolated paper-trading scenario (a COHORT).

    Holds its OWN capital, equity, peak, drawdown, open positions, closed
    trades, equity curve and cost accounting, plus a `market_filter` (a set of
    allowed internal market codes, or None for all markets). Every cohort shares
    the engine's single data fetch, one FinBERT snapshot and one learning brain;
    only the money and the book are private. A lone DISTRIBUTED portfolio
    (market_filter=None) reproduces the engine's original single-portfolio
    behaviour byte-for-byte."""

    def __init__(self, pid: str, name: str, initial_capital: float,
                 market_filter: set[str] | None = None):
        self.id = pid
        self.name = name
        self.market_filter = set(market_filter) if market_filter else None
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.drawdown_pct = 0.0
        self.cost_drag_total = 0.0
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict] = []
        self.equity_curve: list[dict] = []
        # Per-cohort module tracking (the shared brain still learns from all).
        self.agent_stats = {
            "trend_follower": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
            "mean_reverter": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
            "breakout": {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0},
        }
        # Per-cohort transient decision state (keyed by symbol — unique within
        # a cohort's own book even when peers hold the same symbol).
        self._verdicts: dict[str, object] = {}
        self._entry_votes: dict[str, list] = {}
        self._position_consensus: dict[str, dict] = {}
        # Anti-churn: symbol -> UTC datetime it was last closed (re-entry cooldown).
        self._last_closed: dict[str, datetime] = {}

    def allows(self, market: str) -> bool:
        """True if this cohort is permitted to trade the given internal market."""
        return self.market_filter is None or market in self.market_filter

    def market_label(self) -> str:
        return "all" if self.market_filter is None else "+".join(sorted(self.market_filter))

    def next_trade_id(self) -> str:
        return f"T{len(self.closed_trades) + 1:04d}"

    def summary(self) -> dict:
        """Compact snapshot for the dashboard SCOREBOARD / /api/cohorts."""
        closed = [t for t in self.closed_trades
                  if t.get("pnl") is not None]
        unrealized = sum(p.unrealized for p in self.positions.values())
        return {
            "id": self.id,
            "name": self.name,
            "market_filter": sorted(self.market_filter) if self.market_filter else ["all"],
            "market_label": self.market_label(),
            "capital": round(self.initial_capital, 4),
            "equity": round(self.equity, 4),
            "equity_mark": round(self.equity + unrealized, 4),
            "unrealized_pnl": round(unrealized, 4),
            "peak_equity": round(self.peak_equity, 4),
            "pnl": round(self.equity - self.initial_capital, 4),
            "return_pct": round((self.equity - self.initial_capital)
                                / self.initial_capital * 100, 2)
            if self.initial_capital else 0.0,
            "drawdown": round(self.drawdown_pct, 4),
            "open_positions": len(self.positions),
            "closed_trades": len(closed),
            "cost_drag": round(self.cost_drag_total, 4),
            "equity_curve": [
                {"date": p.get("date"), "equity": p.get("equity")}
                for p in self.equity_curve[-120:]
            ],
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
        # The Partners' Room — 10 deterministic investor personas + managers
        self.personas = PersonaEngine(brain=self.brain)
        self.risk_manager = RiskManagerAgent()
        self.portfolio_manager = PortfolioManagerAgent(stats=self.personas.stats)
        # Live-tunable parameters (module constants are just the defaults;
        # the dashboard's Editor's Desk mutates these through apply_config).
        # initial_capital is a property delegating to the primary cohort (seeded
        # at INITIAL_CAPITAL above); only the shared, engine-level knobs live here.
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

        # ── COHORTS: one or more isolated paper portfolios (see parse_cohorts).
        # A single DISTRIBUTED cohort (COHORTS unset) == the original behaviour.
        specs = parse_cohorts(os.environ.get("COHORTS"))
        self.portfolios: dict[str, Portfolio] = {}
        for name, market_filter in specs:
            pid = name
            self.portfolios[pid] = Portfolio(
                pid, name, INITIAL_CAPITAL, market_filter)
        # The "primary" cohort drives every legacy top-level dashboard panel so
        # nothing renders differently in single-portfolio mode: prefer
        # DISTRIBUTED, else the first cohort declared.
        self.primary: Portfolio = (
            self.portfolios.get("DISTRIBUTED")
            or next(iter(self.portfolios.values())))

        # ── The daily improvement loop (Karpathy keep-or-revert) ──
        self._anchors_frozen = True
        self.last_outer_run: str | None = None      # ISO ts of last daily loop
        self._last_datasets: dict[str, pd.DataFrame] = {}  # freshest OHLCV for holdout
        self.loop_best_metric: float | None = None
        self.loop_holdout_sharpe: float | None = None
        self.loop_open_hypotheses: list[str] = []
        self.memory: list[dict] = []
        self.cycle = 0
        self.running = True
        self.paused = False
        self.last_scan_at: str | None = None
        self._scan_now_event = asyncio.Event()

    # ── Backward-compat: top-level engine state proxies the PRIMARY cohort ──
    # so every legacy method / dashboard panel that reads self.equity, etc.
    # keeps working unchanged in single-portfolio mode.
    @property
    def equity(self): return self.primary.equity
    @equity.setter
    def equity(self, v): self.primary.equity = v

    @property
    def peak_equity(self): return self.primary.peak_equity
    @peak_equity.setter
    def peak_equity(self, v): self.primary.peak_equity = v

    @property
    def drawdown_pct(self): return self.primary.drawdown_pct
    @drawdown_pct.setter
    def drawdown_pct(self, v): self.primary.drawdown_pct = v

    @property
    def cost_drag_total(self): return self.primary.cost_drag_total
    @cost_drag_total.setter
    def cost_drag_total(self, v): self.primary.cost_drag_total = v

    @property
    def initial_capital(self): return self.primary.initial_capital
    @initial_capital.setter
    def initial_capital(self, v): self.primary.initial_capital = v

    @property
    def positions(self): return self.primary.positions

    @property
    def closed_trades(self): return self.primary.closed_trades
    @closed_trades.setter
    def closed_trades(self, v): self.primary.closed_trades = v

    @property
    def equity_curve(self): return self.primary.equity_curve
    @equity_curve.setter
    def equity_curve(self, v): self.primary.equity_curve = v

    @property
    def agent_stats(self): return self.primary.agent_stats

    @property
    def _verdicts(self): return self.primary._verdicts

    @property
    def _entry_votes(self): return self.primary._entry_votes

    @property
    def _position_consensus(self): return self.primary._position_consensus

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

    def _engine_state_row(self, portfolio: "Portfolio | None" = None) -> dict:
        p = portfolio or self.primary
        return {
            "portfolio_id": p.id,
            "equity": p.equity,
            "peak_equity": p.peak_equity,
            "initial_capital": p.initial_capital,
            "target_equity": self.target_equity,
            "cycle": self.cycle,
            "params": {
                "name": p.name,
                "market_filter": (sorted(p.market_filter)
                                  if p.market_filter else None),
                "scan_interval_minutes": self.scan_interval_minutes,
                "max_concurrent": self.max_concurrent,
                "max_position_pct": self.max_position_pct,
                "scan_timeframe": self.scan_timeframe,
                "sentry_interval_seconds": self.sentry_interval_seconds,
                "trail_activation_atr": self.trail_activation_atr,
                "trail_distance_atr": self.trail_distance_atr,
                "paused": self.paused,
                "cost_drag_total": p.cost_drag_total,
                "last_outer_run": self.last_outer_run,
                "agent_stats": p.agent_stats,
            },
        }

    def _persist_engine_state(self):
        """Fire-and-forget snapshot of every cohort's engine_state row."""
        if self.store.enabled:
            for p in self.portfolios.values():
                self.store.fire(self.store.save_engine_state(self._engine_state_row(p)))

    def _restore_portfolio(self, portfolio: "Portfolio", st: dict):
        """Apply a persisted engine_state row to one cohort. Engine-level
        (shared) knobs are taken from the primary cohort's row only."""
        portfolio.equity = float(st["equity"])
        portfolio.peak_equity = float(st["peak_equity"])
        portfolio.initial_capital = float(st["initial_capital"])
        portfolio.drawdown_pct = (
            (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity * 100
            if portfolio.peak_equity > 0 else 0.0)
        params = st.get("params") or {}
        portfolio.cost_drag_total = float(
            params.get("cost_drag_total", portfolio.cost_drag_total))
        saved_agents = params.get("agent_stats") or {}
        for name, stats in saved_agents.items():
            portfolio.agent_stats[name] = stats
        if portfolio is self.primary:
            self.target_equity = float(st["target_equity"])
            self.cycle = int(st["cycle"])
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
            self.last_outer_run = params.get("last_outer_run", self.last_outer_run)

    async def restore_state(self):
        """Connect the store and resume every cohort from Postgres."""
        await self.store.init()
        if not self.store.enabled:
            self._mem("Persistence: DATABASE_URL not set — local JSONL/file "
                      "fallback only (state resets on restart)")
            return

        rows = await self.store.load_engine_states()
        by_id = {r["portfolio_id"]: r for r in rows}
        any_restored = False
        for pid, portfolio in self.portfolios.items():
            st = by_id.get(pid)
            if st:
                self._restore_portfolio(portfolio, st)
                any_restored = True

        # Open positions, per cohort
        for p in await self.store.load_positions():
            pid = p.get("portfolio_id") or "DISTRIBUTED"
            portfolio = self.portfolios.get(pid)
            if portfolio is None:
                continue
            # DB rows don't carry a `market` column (yet); derive it from the
            # already-correctly-persisted currency + symbol shape rather than
            # defaulting to "us" (which would reintroduce the mis-costing bug
            # for a restored NSE position after a restart).
            restored_currency = p.get("currency") or "USD"
            if restored_currency == "INR":
                restored_market = "india"
            elif "/" in p["symbol"]:
                restored_market = "crypto"
            else:
                restored_market = "us"
            portfolio.positions[p["symbol"]] = Position(
                p["symbol"], p["side"], float(p["entry_price"]),
                float(p["stop_loss"]), float(p["take_profit"]),
                float(p["size"]), p.get("module") or "unknown",
                p.get("opened_at"), list(p.get("reasons") or []),
                atr=float(p.get("atr") or 0.0),
                currency=restored_currency,
                fx_rate=float(p.get("fx_rate") or 1.0),
                market=restored_market,
            )

        # Journal + curve, per cohort; memory is shared
        for pid, portfolio in self.portfolios.items():
            portfolio.closed_trades = await self.store.load_trades(
                limit=1000, portfolio_id=pid)
            portfolio.equity_curve = await self.store.load_equity_curve(
                limit=500, portfolio_id=pid)
        db_memory = await self.store.load_memory(limit=100)
        if db_memory:
            self.memory = db_memory + self.memory

        # The learning brain reads its own (shared) history from the DB
        await self.brain.stats.load_from_db()
        await self.attribution.load_regime_perf_from_db()

        if any_restored:
            self._mem(
                f"RESUMED from Postgres: cycle {self.cycle}, "
                f"{len(self.portfolios)} cohort(s); primary "
                f"{self.primary.name} equity ${self.equity:.2f} "
                f"(peak ${self.peak_equity:.2f}), "
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
                      f"${cap:.2f} on cohort {self.primary.name} "
                      f"(equity now ${self.equity:.2f})")

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

    def currency_of(self, symbol: str) -> str:
        """Native quote currency for a symbol. NSE quotes in INR; US equities
        and crypto (USDT pairs) are priced in USD."""
        return "INR" if self.market_of(symbol) == "india" else "USD"

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
        elif hours.get("india_holiday"):
            _nm = hours.get("india_holiday_name")
            self._mem(f"NSE holiday{f' ({_nm})' if _nm else ''} — "
                      "Indian equities closed today")
        else:
            self._mem("NSE closed — skipping Indian equities this scan")
        if hours["us"]:
            fetch_markets.append("us")
        elif hours.get("us_holiday"):
            _nm = hours.get("us_holiday_name")
            self._mem(f"NYSE holiday{f' ({_nm})' if _nm else ''} — "
                      "US equities closed today")
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

    def score_asset(self, portfolio: "Portfolio", symbol: str, df: pd.DataFrame,
                    macro: dict | None = None) -> dict | None:
        # Scoring runs PER cohort: the technical/indicator/persona maths is
        # deterministic on the shared candle data, but the cost-aware gate and
        # the RuleBrain context read THIS cohort's own equity/book, so a setup
        # can pass for one portfolio and be cost-rejected for another.
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

        # Module decided FIRST, so the target can be sized to actually clear
        # the immutable rule that gates THAT module.
        #
        # Bug this replaces: SL/TP used to be a single fixed 2x/4.2x ATR ratio
        # (always exactly 2.1:1) computed before classification. D4
        # (Druckenmiller, IMMUTABLE) requires >=3:1 on the trend_follower
        # sleeve specifically -- so every trend_follower-classified setup
        # (the highest-conviction |score|>50 signals) was being vetoed 100%
        # of the time, everywhere, forever. That is a structural lockout, not
        # the rule "correctly filtering low quality" -- 2.1 can never reach
        # 3.0 no matter how good the setup is. Fix: widen ONLY the
        # trend_follower target so it can clear D4 with margin; leave
        # breakout/mean_reverter at the existing 2.1:1 (already clears the
        # separate 2:1 floor those modules are actually gated on).
        direction = "BUY" if score > 0 else "SELL"
        module = "trend_follower" if abs(score) > 50 else "breakout" if vr > 1.3 else "mean_reverter"
        tp_mult = 6.4 if module == "trend_follower" else 4.2  # -> 3.2:1 vs 2.1:1
        if direction == "BUY":
            sl = price - 2 * at
            tp = price + tp_mult * at
        else:
            sl = price + 2 * at
            tp = price - tp_mult * at

        rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

        # ── Cost-aware entry gate (applies ALONGSIDE the immutable R:R>=2:1) ──
        # Uses the PER-MARKET round-trip cost at the SIZE this trade would take
        # (planned notional = max_position_pct of equity). For NSE the flat Rs.20
        # per-order floor dominates a tiny notional, so most $10 NSE setups are
        # correctly rejected; US (~0.06%) and crypto (~0.30%) setups pass.
        # ── Anti-churn re-entry cooldown ──────────────────────────────────
        # After closing a name, don't immediately re-open it — the biggest
        # source of fee bleed is the 15m loop round-tripping the same symbol.
        last_close = portfolio._last_closed.get(symbol)
        if last_close is not None and REENTRY_COOLDOWN_MIN > 0:
            age_min = (datetime.now(timezone.utc) - last_close).total_seconds() / 60.0
            if age_min < REENTRY_COOLDOWN_MIN:
                log.info(f"SKIP {symbol} [{portfolio.id}]: re-entry cooldown "
                         f"({age_min:.0f}m < {REENTRY_COOLDOWN_MIN:g}m since last close)")
                return None

        # Resolve market/currency ONCE here and thread them through the rest of
        # this scan + the setup dict — never re-derive via self.market_of() /
        # self.currency_of() later (those read the shared self.symbol_markets
        # dict, which a concurrent scan can rebuild between scoring and
        # open_position(), silently flipping currency and mis-sizing ~83x).
        market = self.market_of(symbol)
        currency = "INR" if market == "india" else "USD"
        fx_est = usd_rate(currency)
        planned_notional_usd = portfolio.equity * self.max_position_pct / 100
        rt_cost_usd = round_trip_cost_usd(market, planned_notional_usd, fx_est)
        rt_cost_pct = round_trip_cost_pct(market, planned_notional_usd, fx_est)
        tp_move_pct = abs(tp - price) / price * 100 if price else 0.0
        cost_floor = COST_EDGE_MULTIPLE * rt_cost_pct
        if tp_move_pct < cost_floor:
            flat_note = "flat-fee " if market == "india" else ""
            log.info(f"SKIP {symbol} [{portfolio.id}]: {flat_note}cost {rt_cost_pct:.2f}% "
                     f"round-trip at ${planned_notional_usd:.0f} notional exceeds edge — "
                     f"TP move {tp_move_pct:.2f}% < floor {cost_floor:.2f}% "
                     f"({COST_EDGE_MULTIPLE:g}x)")
            self.audit.log_trade_decision(
                action="SKIP", symbol=symbol,
                setup={"symbol": symbol, "direction": direction,
                       "current_price": round(price, 2),
                       "take_profit": round(tp, 2), "risk_reward": round(rr, 2)},
                portfolio_context={"equity": round(portfolio.equity, 4),
                                   "portfolio_id": portfolio.id},
                reasoning=(f"Cost-aware veto ({market}): target {tp_move_pct:.2f}% < "
                           f"{cost_floor:.2f}% floor ({COST_EDGE_MULTIPLE:g}x "
                           f"round-trip {rt_cost_pct:.2f}% at ${planned_notional_usd:.0f})."),
            )
            return None

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
            "market": market,
            # Stamp currency/fx HERE, from the market context this scan already
            # resolved locally — never re-derive it later from the shared,
            # per-cycle-mutable self.symbol_markets dict (that's a race: a
            # concurrent scan-now / scheduled scan can rebuild that dict between
            # scoring and open_position(), silently flipping an NSE symbol's
            # currency to the "us" default and mis-sizing the position ~83x).
            "currency": currency,
            "fx_rate": fx_est,
            "est_round_trip_cost_usd": round(rt_cost_usd, 4),
            "est_round_trip_cost_pct": round(rt_cost_pct, 3),
            "tp_move_pct": round(tp_move_pct, 3),
        }

        # ── Consult the RuleBrain: every decision carries rule citations ──
        ctx = self._rule_context(portfolio, result)
        verdict = self.brain.evaluate_setup(result, ctx)
        portfolio._verdicts[symbol] = verdict
        result["rule_citations"] = verdict.citations
        result["rule_failures"] = [c.citation for c in verdict.failed]
        result["rule_score_multiplier"] = verdict.score_multiplier

        if verdict.vetoed:
            # IMMUTABLE risk rule failed — hard veto, non-negotiable
            self.audit.log_rule_citations(symbol, "VETO", verdict.citations, verdict.veto_citations)
            self.audit.log_trade_decision(
                action="REJECT", symbol=symbol, setup=result,
                portfolio_context={"equity": round(portfolio.equity, 4),
                                   "drawdown_pct": round(portfolio.drawdown_pct, 4),
                                   "open_positions": len(portfolio.positions),
                                   "portfolio_id": portfolio.id},
                reasoning="IMMUTABLE rule veto: " + " || ".join(verdict.veto_citations),
            )
            self._mem(f"VETO {symbol} [{portfolio.id}]: {verdict.veto_citations[0]}", "FAIL")
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
                    portfolio_context={"equity": round(portfolio.equity, 4),
                                       "open_positions": len(portfolio.positions),
                                       "portfolio_id": portfolio.id},
                    reasoning=f"Persona consensus reversed the signal: {synthesis['summary']}",
                )
                return None
            result["score"] = final
            result["confidence"] = min(1.0, abs(final) / 100)
            if abs(final) < 25:
                return None

        return result

    def _rule_context(self, portfolio: "Portfolio", setup: dict) -> dict:
        """Portfolio/market context handed to the RuleBrain evaluators — read
        from THIS cohort's own book so each portfolio is judged independently."""
        price = setup["current_price"]
        stop_pct = abs(price - setup["stop_loss"]) / price * 100 if price else 0.0
        # Planned sizing mirrors open_position(): notional capped at max_position_pct,
        # so realized risk = notional% x stop%.
        planned_notional_pct = self.max_position_pct
        planned_risk_pct = planned_notional_pct * stop_pct / 100.0

        gross = 0.0
        if portfolio.equity > 0:
            gross = sum(p.entry_price * p.size for p in portfolio.positions.values()) / portfolio.equity * 100
        closed = [t for t in portfolio.closed_trades if t.get("pnl") is not None]
        last10 = closed[-10:]
        wins10 = sum(1 for t in last10 if (t.get("pnl") or 0) > 0)
        pos = portfolio.positions.get(setup["symbol"])

        return {
            "equity": portfolio.equity,
            "drawdown_pct": portfolio.drawdown_pct,
            "total_return_pct": (portfolio.equity - portfolio.initial_capital) / portfolio.initial_capital * 100,
            "gross_exposure_pct": gross,
            "planned_notional_pct": planned_notional_pct,
            "planned_risk_pct": planned_risk_pct,
            "open_positions": len(portfolio.positions),
            "max_concurrent": self.max_concurrent,
            "last10_win_rate": wins10 / len(last10) if last10 else None,
            "last10_n": len(last10),
            "existing_position_unrealized": pos.unrealized if pos else None,
            "module_stats": portfolio.agent_stats.get(setup["module"]),
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

    def check_positions(self, portfolio: "Portfolio", datasets: dict[str, pd.DataFrame]):
        """Scout-side SL/TP/trailing check on the freshest candle (wick-accurate
        via high/low). Sentry protects between scans; both pop from the same
        dict so a position can never be double-closed."""
        to_close = []
        for sym, pos in portfolio.positions.items():
            if sym not in datasets:
                continue
            df = datasets[sym]
            last = df.iloc[-1]
            high = float(last["high"])
            low = float(last["low"])
            close = float(last["close"])

            # Update unrealized — native price move / fx = TRUE USD P&L
            fx = pos.fx_rate or 1.0
            if pos.side == "BUY":
                pos.unrealized = (close - pos.entry_price) * pos.size / fx
            else:
                pos.unrealized = (pos.entry_price - close) * pos.size / fx

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
            self._close_position(portfolio, sym, exit_price, reason)

    def _close_position(self, portfolio: "Portfolio", symbol: str,
                        exit_price: float, reason: str):
        pos = portfolio.positions.pop(symbol, None)
        if not pos:
            return

        # Stamp close time for the anti-churn re-entry cooldown.
        portfolio._last_closed[symbol] = datetime.now(timezone.utc)

        # Per-market exit cost (brokerage + slippage + tax) in TRUE USD. Fill at
        # the native mid — slippage is captured inside the cost model. Use the
        # market/fx captured on THIS position at open time, not a fresh
        # self.market_of() lookup — the shared symbol_markets dict can have
        # been rebuilt (different universe) in the hours since this position
        # was opened, which would otherwise apply the wrong market's fee model.
        fx = pos.fx_rate or 1.0
        market = getattr(pos, "market", None) or self.market_of(symbol)
        notional_usd = pos.size * exit_price / fx
        exit_cost = order_cost_usd(market, notional_usd, fx)

        # Gross P&L is a native price move x size; divide by fx for TRUE USD.
        # pnl is booked NET of this exit cost (entry cost was taken on open).
        if pos.side == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.size / fx - exit_cost
        else:
            pnl = (pos.entry_price - exit_price) * pos.size / fx - exit_cost

        portfolio.equity += pnl
        portfolio.peak_equity = max(portfolio.peak_equity, portfolio.equity)
        portfolio.drawdown_pct = (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity * 100
        # Cost bookkeeping (TRUE USD). pnl above is already NET of exit_cost.
        portfolio.cost_drag_total += exit_cost

        trade = {
            "id": portfolio.next_trade_id(),
            "portfolio_id": portfolio.id,
            "date": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "side": pos.side,
            # entry/exit shown in NATIVE currency; pnl is TRUE USD.
            "entry": pos.entry_price, "exit": round(exit_price, 2),
            "size": pos.size, "pnl": round(pnl, 4),
            "reason": reason, "module": pos.module,
            "currency": pos.currency, "fx_rate": round(fx, 4),
            "est_round_trip_cost_usd": round(pos.est_round_trip_cost_usd, 4),
            "est_round_trip_cost_pct": round(pos.est_round_trip_cost_pct, 3),
            "exit_cost_usd": round(exit_cost, 4),
        }
        portfolio.closed_trades.append(trade)
        if self.store.enabled:
            self.store.fire(self.store.delete_position(symbol, portfolio.id))

        # Agent stats (this cohort's own module ledger)
        stats = portfolio.agent_stats.get(pos.module, {"wins":0,"losses":0,"pnl":0,"trades":0})
        stats["trades"] += 1
        stats["pnl"] += pnl
        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        portfolio.agent_stats[pos.module] = stats

        # Audit — analyst-grade outcome log
        self.audit.log_trade_outcome(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=round(exit_price, 2),
            pnl=round(pnl, 4), exit_reason=reason, duration_hours=0,
            entry_reasons=pos.reasons,
            market_at_exit={"rsi": 0, "adx": 0},  # filled from latest data when available
        )

        # Attribution — decompose P&L (§9.1) and update per-rule win/loss stats.
        # Feed USD-space entry/exit so the headline thesis/timing decomposition
        # matches the TRUE-USD pnl (fx=1.0 for USD assets — unchanged there).
        # trade_id is namespaced by cohort so peers holding the same symbol keep
        # independent open-attribution snapshots (the derived regime/rule stats
        # still feed the ONE shared brain).
        attribution = self.attribution.on_trade_close(
            trade_id=f"{portfolio.id}::{symbol}",
            exit_price=round(exit_price / fx, 6), exit_reason=reason,
            pnl=round(pnl, 4), size=pos.size, entry_price=pos.entry_price / fx,
        )
        if attribution:
            self._mem(
                f"ATTRIBUTION {symbol} [{portfolio.id}]: thesis ${attribution.thesis_pnl:+.4f}, "
                f"execution ${attribution.timing_pnl:+.4f}, regime={attribution.regime}",
                "info",
            )

        # Partners' Room bookkeeping: score each persona's entry vote against
        # the realized outcome (accuracy feeds the PM's vote weights).
        entry_votes = portfolio._entry_votes.pop(symbol, None)
        portfolio._position_consensus.pop(symbol, None)
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

        # For non-USD assets, spell out the currency fix: the same move is worth
        # fx-times fewer real dollars than the raw native-point number.
        fx_note = ""
        if pos.currency != "USD" and fx != 1.0:
            gross_native = ((exit_price - pos.entry_price) if pos.side == "BUY"
                            else (pos.entry_price - exit_price)) * pos.size
            fx_note = (f" [{pos.currency} {gross_native:+.2f} pts @ {fx:.2f} "
                       f"{pos.currency}/USD = ${gross_native / fx:+.4f} USD]")
        if pnl > 0:
            self._mem(f"WIN ${pnl:+.4f} on {symbol} [{portfolio.id}] ({reason}).{fx_note} "
                      f"{', '.join(pos.reasons[:2])}", "SUCCESS")
        else:
            self._mem(f"LOSS ${pnl:+.4f} on {symbol} [{portfolio.id}] ({reason}).{fx_note} "
                      f"Revisit: {', '.join(pos.reasons[:2])}", "FAIL")

    def open_position(self, portfolio: "Portfolio", setup: dict):
        if setup["symbol"] in portfolio.positions:
            return
        if len(portfolio.positions) >= self.max_concurrent:
            return

        price = setup["current_price"]          # native currency (INR for NSE)
        sl = setup["stop_loss"]                  # native currency
        risk_per_unit = abs(price - sl)
        if risk_per_unit <= 0:
            return

        # ── Currency: size and cap notional in TRUE USD ──────────────────
        # equity is USD; NSE prices are INR. Convert price/stop to USD so
        # "10% of equity notional" and per-unit risk are real dollars, not
        # rupee-points. USD assets carry fx=1.0 -> identical to before.
        #
        # Trust the currency/fx the SCAN stamped into setup (resolved once,
        # locally, at score time). Do NOT re-derive via self.currency_of()
        # here — that reads the shared, per-cycle-mutable self.symbol_markets
        # dict, which a concurrent scan can rebuild between scoring and this
        # call, silently flipping an NSE symbol's currency to the "us"
        # default and mis-sizing the position by ~83x (the fx rate). Fall
        # back to the live lookup only for any legacy caller that doesn't
        # pass a pre-stamped setup.
        currency = setup.get("currency") or self.currency_of(setup["symbol"])
        fx = float(setup.get("fx_rate") or usd_rate(currency))
        price_u = price / fx
        risk_per_unit_u = risk_per_unit / fx

        # Size: risk max_position_pct of equity (all USD)
        risk_amount = portfolio.equity * self.max_position_pct / 100
        size = risk_amount / risk_per_unit_u
        notional = size * price_u
        max_notional = portfolio.equity * self.max_position_pct / 100
        if notional > max_notional:
            size = max_notional / price_u

        # Per-market entry cost (brokerage + slippage + tax), booked in TRUE USD.
        # Fill at the native mid; the cost model already accounts for slippage,
        # so we don't also move the fill price (avoids double-counting).
        # Same rule as currency above: trust the scan-stamped market.
        market = setup.get("market") or self.market_of(setup["symbol"])
        entry = price
        notional_usd = size * entry / fx
        entry_cost = order_cost_usd(market, notional_usd, fx)
        portfolio.equity -= entry_cost
        portfolio.cost_drag_total += entry_cost
        rt_cost_usd = round_trip_cost_usd(market, notional_usd, fx)
        rt_cost_pct = round_trip_cost_pct(market, notional_usd, fx)

        pos = Position(
            setup["symbol"], setup["direction"], round(entry, 2),
            sl, setup["take_profit"], round(size, 6),
            setup["module"], datetime.now(timezone.utc).isoformat(),
            setup["reasons"], atr=float(setup.get("atr") or 0.0),
            currency=currency, fx_rate=fx, market=market,
        )
        pos.est_round_trip_cost_usd = round(rt_cost_usd, 4)
        pos.est_round_trip_cost_pct = round(rt_cost_pct, 3)
        portfolio.positions[setup["symbol"]] = pos

        open_row = {
            "id": portfolio.next_trade_id(),
            "portfolio_id": portfolio.id,
            "date": datetime.now(timezone.utc).isoformat(),
            "symbol": setup["symbol"], "side": setup["direction"],
            "entry": round(entry, 2), "exit": None,
            "size": round(size, 6), "pnl": None,
            "reason": "OPEN", "module": setup["module"],
            "currency": currency, "fx_rate": round(fx, 4),
            "est_round_trip_cost_usd": round(rt_cost_usd, 4),
            "est_round_trip_cost_pct": round(rt_cost_pct, 3),
        }
        portfolio.closed_trades.append(open_row)

        # Persist: open position row, OPEN journal row (with rule citations),
        # and the post-commission equity snapshot.
        if self.store.enabled:
            self.store.fire(self.store.upsert_position(
                pos.to_dict() | {"reasons": pos.reasons,
                                 "portfolio_id": portfolio.id}))
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
        portfolio._entry_votes[setup["symbol"]] = list(setup.get("persona_votes", []))
        cons = setup.get("persona_consensus")
        if cons:
            portfolio._position_consensus[setup["symbol"]] = dict(cons) | {
                "at": datetime.now(timezone.utc).isoformat()}
            self.audit.log_system_event("PERSONA_CONSENSUS", {
                "symbol": setup["symbol"], "action": cons.get("action"),
                "consensus": cons.get("consensus"), "dissent": cons.get("dissent"),
                "vetoed": cons.get("vetoed"),
            })

        # Audit — full decision reasoning + investor-rule citations.
        # Notional and risk are booked in TRUE USD (native / fx).
        risk_amount = size * abs(entry - sl) / fx
        portfolio_context = {
            "portfolio_id": portfolio.id,
            "equity": round(portfolio.equity, 4),
            "drawdown_pct": round(portfolio.drawdown_pct, 4),
            "open_positions": len(portfolio.positions),
            "position_size": round(size * entry / fx, 4),
            "risk_amount": round(risk_amount, 4),
            "risk_pct": round(risk_amount / portfolio.equity * 100, 2),
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

        # Attribution — store the causal record (§9.1) for close-time decomposition.
        # trade_id namespaced by cohort (matches on_trade_close).
        self.attribution.record_entry(
            trade_id=f"{portfolio.id}::{setup['symbol']}", setup=setup,
            context=portfolio_context | {"equity": portfolio.equity},
            verdict=portfolio._verdicts.get(setup["symbol"]),
        )

        self._mem(
            f"OPEN {setup['direction']} {setup['symbol']} @ ${entry:.2f} "
            f"[{portfolio.id}] SL=${sl} TP=${setup['take_profit']} "
            f"({setup['module']}, score={setup['score']})",
            "info"
        )

    def _cost_by_market_snapshot(self, portfolio: "Portfolio | None" = None) -> dict:
        """Indicative per-market round-trip cost at the current planned notional
        (max_position_pct of equity). Shows why a small account should favour
        US/crypto over flat-fee-dominated NSE intraday."""
        notional = (portfolio or self.primary).equity * self.max_position_pct / 100
        out = {}
        for mkt, cur in (("us", "USD"), ("crypto", "USD"), ("india", "INR")):
            fx = usd_rate(cur)
            out[mkt] = {
                "notional_usd": round(notional, 2),
                "round_trip_cost_usd": round(round_trip_cost_usd(mkt, notional, fx), 4),
                "round_trip_cost_pct": round(round_trip_cost_pct(mkt, notional, fx), 3),
            }
        return out

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
        # THE SCOREBOARD — every cohort side by side (winner spotted at a glance).
        cohorts = [p.summary() for p in self.portfolios.values()]
        state = {
            "cohorts": cohorts,
            "cohort_count": len(cohorts),
            "primary_cohort": self.primary.id,
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
            "cost_drag_total": round(self.cost_drag_total, 4),
            "cost_by_market": self._cost_by_market_snapshot(),
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
            "loop_status": ENGINE_STATE.get("loop_status", {
                "last_outer_run": self.last_outer_run,
                "best_metric": self.loop_best_metric,
                "holdout_sharpe": self.loop_holdout_sharpe,
                "open_hypotheses": self.loop_open_hypotheses,
                "anchors_frozen": self._anchors_frozen,
                "metric_definition": ("Out-of-sample walk-forward Sharpe "
                                      "(last 20% holdout) w/ hard 2% max-DD veto"),
            }),
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
        self._last_datasets = datasets  # freshest OHLCV for the held-out metric
        self._mem(f"Loaded {len(datasets)} assets")

        # 2-4. COHORTS: every portfolio sees the SAME shared datasets this
        # instant, but checks its own book, applies its own market_filter, and
        # runs its OWN ranking/sizing/cost-gate/risk independently. The primary
        # (DISTRIBUTED) cohort drives the legacy top-level dashboard panels.
        primary_scan_results: list[dict] = []
        for portfolio in self.portfolios.values():
            if portfolio.positions:
                self.check_positions(portfolio, datasets)

            setups = []
            scan_results = []
            for sym, df in datasets.items():
                if not portfolio.allows(self.market_of(sym)):
                    continue
                result = self.score_asset(portfolio, sym, df, macro=self.macro_snapshot)
                if result:
                    scan_results.append(result)
                    if abs(result["score"]) >= 25 and sym not in portfolio.positions:
                        setups.append(result)

            # Rank and execute for THIS cohort
            if setups and len(portfolio.positions) < self.max_concurrent:
                ranked = sorted(
                    setups,
                    key=lambda x: abs(x["score"]) * x["confidence"] * min(x["risk_reward"], 3),
                    reverse=True)
                slots = self.max_concurrent - len(portfolio.positions)
                for setup in ranked[:slots]:
                    self.open_position(portfolio, setup)
                for setup in ranked[slots:]:
                    self.audit.log_trade_decision(
                        action="SKIP", symbol=setup["symbol"], setup=setup,
                        portfolio_context={"equity": portfolio.equity,
                                           "open_positions": len(portfolio.positions),
                                           "portfolio_id": portfolio.id},
                        reasoning=(f"Ranked lower than selected trades. "
                                   f"Score={setup['score']}, max slots filled."))

            # Per-cohort equity point (drives the SCOREBOARD sparklines)
            eq_point = {
                "date": datetime.now(timezone.utc).isoformat(),
                "equity": round(portfolio.equity, 4),
                "drawdown_pct": round(portfolio.drawdown_pct, 4),
                "positions": len(portfolio.positions),
            }
            portfolio.equity_curve.append(eq_point)
            if self.store.enabled:
                self.store.fire(self.store.log_equity_point(eq_point, portfolio.id))

            self._mem(f"[{portfolio.id}] {len(setups)} actionable of "
                      f"{len(scan_results)} scanned; {len(portfolio.positions)} open")
            if portfolio is self.primary:
                primary_scan_results = scan_results

        # The dashboard's top-level panels mirror the primary cohort's scan.
        scan_results = primary_scan_results
        setups = [r for r in scan_results
                  if abs(r.get("score", 0)) >= 25
                  and r["symbol"] not in self.primary.positions]

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

        # Log the primary cohort's scans to audit (ranking/opening already
        # happened per-cohort in the loop above).
        for sr in scan_results:
            self.audit.log_scan(sr["symbol"], sr)

        # 5. Durable snapshot of every cohort's state (Postgres when configured)
        if self.store.enabled:
            self._persist_engine_state()

        # 6. Push to dashboard (last_scan anchors the client-side countdown)
        self.last_scan_at = datetime.now(timezone.utc).isoformat()
        await self.push_state()

        # 6b. The daily improvement loop — keep-or-revert against the confirmed
        # held-out metric, at most once per 24h of wall-clock.
        if self._daily_loop_due():
            try:
                await self.run_daily_outer_loop()
            except Exception as e:
                self._mem(f"Daily loop error: {e}", "FAIL")
                log.error(traceback.format_exc())

        # 7. Check target
        if self.equity >= self.target_equity:
            self._mem(f"TARGET HIT! Equity ${self.equity:.2f} >= ${self.target_equity:.2f}", "SUCCESS")
            self.running = False

    # ── The daily improvement loop (Karpathy keep-or-revert) ────────

    def _daily_loop_due(self) -> bool:
        """True if >= 24h of wall-clock has passed since the last outer run."""
        if not self.last_outer_run:
            return True
        try:
            last = datetime.fromisoformat(self.last_outer_run)
        except (TypeError, ValueError):
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last) >= timedelta(hours=24)

    def _loop_config_snapshot(self) -> dict:
        """The current live-tunable configuration (what a 'config' means here)."""
        return {
            "scan_timeframe": self.scan_timeframe,
            "max_position_pct": self.max_position_pct,
            "max_concurrent": self.max_concurrent,
            "trail_activation_atr": self.trail_activation_atr,
            "trail_distance_atr": self.trail_distance_atr,
            "rule_weights_source": "knowledge/rule_stats.json (live attribution)",
            "learnable_rule_weights": self.brain.stats.snapshot().get("rules", []),
        }

    async def score_holdout_metric(self) -> dict:
        """Score the current config on the CONFIRMED metric: out-of-sample
        walk-forward Sharpe (last 20% holdout) with the 2% max-DD veto.

        Returns a dict; `holdout_sharpe` is None when the veto fires or there is
        no scorable held-out data. `raw_holdout_sharpe` is the mean Sharpe for
        display even when vetoed."""
        from strategy.modules import ALL_MODULES
        from backtest.engine import BacktestEngine, WalkForwardEvaluator

        datasets = self._last_datasets or {}
        if not datasets:
            return {"scorable": False, "reason": "no datasets fetched yet",
                    "holdout_sharpe": None, "raw_holdout_sharpe": None,
                    "vetoed": False, "per_module": [],
                    "assets_evaluated": 0, "total_holdout_trades": 0}

        ev = WalkForwardEvaluator(
            BacktestEngine(risk_config=self.settings.risk,
                           initial_capital=100_000.0),
            holdout_pct=0.20,
            max_drawdown_pct=self.settings.risk.max_drawdown_pct,
        )
        # Cap assets for bounded runtime (daily cadence, but stay responsive).
        items = dict(list(datasets.items())[:15])

        per_module, ok_sharpes, all_sharpes = [], [], []
        vetoed_any = False
        scored = 0
        trades = 0
        for ModuleCls in ALL_MODULES:
            module = ModuleCls()
            s = ev.score(module, items)  # raw score (surfaces Sharpe even on veto)
            if s is None:
                per_module.append({"module": module.name, "scorable": False})
                continue
            scored += 1
            trades += s.total_trades
            all_sharpes.append(s.holdout_sharpe)
            if s.vetoed:
                vetoed_any = True
            else:
                ok_sharpes.append(s.holdout_sharpe)
            per_module.append({
                "module": module.name,
                "holdout_sharpe": s.holdout_sharpe,
                "max_dd_pct": s.holdout_max_dd_pct,
                "return_pct": s.holdout_return_pct,
                "trades": s.total_trades,
                "assets": s.assets_scored,
                "vetoed": s.vetoed,
            })

        raw = round(sum(all_sharpes) / len(all_sharpes), 4) if all_sharpes else None
        if scored == 0:
            agg, reason = None, "no scorable held-out data"
        elif vetoed_any:
            agg, reason = None, "2% max-drawdown veto breached on holdout"
        else:
            agg = round(sum(ok_sharpes) / len(ok_sharpes), 4) if ok_sharpes else None
            reason = "ok"
        return {"scorable": scored > 0, "reason": reason,
                "holdout_sharpe": agg, "raw_holdout_sharpe": raw,
                "vetoed": vetoed_any, "per_module": per_module,
                "assets_evaluated": len(items), "total_holdout_trades": trades}

    @staticmethod
    def _decide_verdict(before, after, vetoed: bool, scorable: bool):
        """Karpathy keep-or-revert against the REAL held-out metric."""
        if vetoed:
            return "VETO", ("Held-out 2% max-drawdown veto breached — config "
                            "rejected on unseen data.")
        if not scorable or after is None:
            return "SEED", ("No scorable held-out data yet — baseline recorded, "
                            "no keep/revert.")
        if before is None:
            return "KEEP", "First real held-out measurement — recorded as best."
        if after >= before - 1e-4:
            return "KEEP", "Held-out Sharpe did not regress — configuration kept."
        return "REVERT", ("Held-out Sharpe regressed vs best — configuration "
                          "flagged for revert (overfit signature).")

    def _load_loop_state(self) -> dict:
        try:
            return json.loads((LOOP_DIR / "state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_loop_state(self, st: dict) -> None:
        (LOOP_DIR).mkdir(parents=True, exist_ok=True)
        (LOOP_DIR / "state.json").write_text(
            json.dumps(st, indent=2), encoding="utf-8")

    def _next_experiment_id(self) -> str:
        d = LOOP_DIR / "experiments"
        d.mkdir(parents=True, exist_ok=True)
        nums = []
        for f in d.glob("*.json"):
            try:
                nums.append(int(f.name.split("-")[0]))
            except (ValueError, IndexError):
                pass
        n = (max(nums) + 1) if nums else 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{n:04d}-{stamp}"

    def _write_experiment(self, record: dict) -> None:
        d = LOOP_DIR / "experiments"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{record['id']}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")

    def _append_lesson(self, exp_id, verdict, before, after, downweighted, vetoed):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ids = ",".join(d["rule_id"] for d in downweighted) or "none"
        if vetoed:
            line = (f"- {day} [{exp_id}] current config -> held-out 2% max-DD "
                    f"VETO -> do not ship; find which module breached DD on the "
                    f"unseen slice before re-scoring.")
        else:
            line = (f"- {day} [{exp_id}] live-weighted config -> held-out Sharpe "
                    f"{after} < best {before} (REVERT); down-weighted {ids} -> a "
                    f"further cut here regressed unseen Sharpe, don't chase it.")
        try:
            with open(LOOP_DIR / "lessons.md", "a", encoding="utf-8") as f:
                f.write("\n" + line)
        except OSError as e:
            log.warning(f"could not append lesson: {e}")

    def _update_loop_status_state(self, score: dict, verdict: str, downweighted: list):
        ENGINE_STATE["loop_status"] = {
            "last_outer_run": self.last_outer_run,
            "best_metric": self.loop_best_metric,
            "holdout_sharpe": score.get("holdout_sharpe"),
            "raw_holdout_sharpe": score.get("raw_holdout_sharpe"),
            "vetoed": score.get("vetoed"),
            "last_verdict": verdict,
            "downweighted_rules": [d["rule_id"] for d in downweighted],
            "open_hypotheses": self.loop_open_hypotheses,
            "anchors_frozen": self._anchors_frozen,
            "metric_definition": ("Out-of-sample walk-forward Sharpe "
                                  "(last 20% holdout) w/ hard 2% max-DD veto"),
            "per_module": score.get("per_module", []),
            "assets_evaluated": score.get("assets_evaluated", 0),
            "holdout_trades": score.get("total_holdout_trades", 0),
        }

    async def run_daily_outer_loop(self, force: bool = False) -> dict:
        """The daily Karpathy keep-or-revert against the REAL metric on FRESH
        held-out data. Reads LIVE attribution + rule_stats, scores the current
        config on the held-out walk-forward Sharpe (2% DD veto), down-weights
        weak LEARNABLE rules, NEVER touches anchors, and records the outcome."""
        if not force and not self._daily_loop_due():
            return {"ran": False, "reason": "not due"}

        # (d) NEVER touch anchors — verify frozen before doing anything.
        try:
            assert_anchors_untouched()
            self._anchors_frozen = True
            self._mem("Anchors verified frozen.", "SUCCESS")
        except AnchorTamperError as e:
            self._anchors_frozen = False
            self._mem(f"Daily loop ABORTED — anchor tamper: {e}", "FAIL")
            raise

        now = datetime.now(timezone.utc).isoformat()
        self._mem("DAILY LOOP: scoring live config against the held-out "
                  "walk-forward Sharpe (2% DD veto)...")

        # (a) LIVE closed-trade attribution + rule_stats (reload file writes).
        self.brain.stats._load()
        weak = self.brain.stats.underperformers(accuracy_below=0.4)
        closed = [t for t in self.closed_trades if t.get("pnl") is not None]

        # (b) score current config on the confirmed held-out metric.
        score = await self.score_holdout_metric()
        metric_after = score["holdout_sharpe"]  # None on veto / no data
        vetoed = score["vetoed"]

        # (c) down-weight LEARNABLE rules hurting the metric. Their weights are
        # already derived from live accuracy in RuleStats (< 1.0 when wrong),
        # which feeds RuleBrain's score_multiplier — this records/audits it.
        downweighted = [
            {"rule_id": w["rule_id"], "investor": w["investor"],
             "accuracy": w["accuracy"], "samples": w["samples"],
             "weight": w["weight"]}
            for w in weak
        ]
        for w in weak:
            self._mem(f"LOOP down-weight LEARNABLE {w['rule_id']} "
                      f"({w['investor']}): acc {w['accuracy']:.0%} over "
                      f"{w['samples']} trades -> weight {w['weight']}")

        # (e) keep-or-revert vs the best-known held-out metric in state.json.
        st = self._load_loop_state()
        metric_before = st.get("best_metric")
        verdict, note = self._decide_verdict(
            metric_before, metric_after, vetoed, score["scorable"])

        snapshot = self._loop_config_snapshot()
        exp_id = self._next_experiment_id()
        record = {
            "id": exp_id,
            "timestamp": now,
            "hypothesis": ("Live rule weights (from attribution) + current params "
                           "maximize held-out walk-forward Sharpe without breaching "
                           "the 2% max-drawdown veto."),
            "diff_summary": (f"{len(downweighted)} LEARNABLE rules down-weighted "
                             f"from live attribution; params unchanged. "
                             f"timeframe={snapshot['scan_timeframe']}, "
                             f"max_pos_pct={snapshot['max_position_pct']}, "
                             f"max_concurrent={snapshot['max_concurrent']}."),
            "metric_before": metric_before,
            "metric_after": metric_after,
            "holdout_score": metric_after,
            "verdict": verdict,
            "verifier_notes": (
                f"{note} | reason={score['reason']} "
                f"assets={score['assets_evaluated']} "
                f"holdout_trades={score['total_holdout_trades']} "
                f"live_closed={len(closed)} "
                f"raw_holdout_sharpe={score['raw_holdout_sharpe']} "
                f"downweighted={[d['rule_id'] for d in downweighted]}"),
            "anchors_frozen": self._anchors_frozen,
            "per_module": score["per_module"],
        }
        self._write_experiment(record)

        # Update state.json (best_* only advance on KEEP) + lessons on regress.
        if verdict == "KEEP":
            st["best_metric"] = metric_after
            st["best_holdout_sharpe"] = metric_after
            st["best_known_config"] = snapshot
        st["last_outer_run"] = now
        st["anchors_frozen"] = self._anchors_frozen
        st.setdefault("open_hypotheses", [])
        self._save_loop_state(st)
        if verdict in ("REVERT", "VETO"):
            self._append_lesson(exp_id, verdict, metric_before, metric_after,
                                downweighted, vetoed)

        # Reflect on the engine + dashboard.
        self.last_outer_run = now
        self.loop_best_metric = st.get("best_metric")
        self.loop_holdout_sharpe = score.get("raw_holdout_sharpe")
        self.loop_open_hypotheses = st.get("open_hypotheses", [])
        self._persist_engine_state()

        # (f) clear summary.
        self._mem(
            f"DAILY LOOP [{verdict}]: held-out Sharpe before={metric_before} "
            f"after={metric_after} (veto={vetoed}); {len(downweighted)} learnable "
            f"rules down-weighted; {len(closed)} live closed trades. {note}",
            "SUCCESS" if verdict == "KEEP" else "info")
        self._update_loop_status_state(score, verdict, downweighted)
        await self.push_state()
        return {"ran": True, "verdict": verdict, "metric_before": metric_before,
                "metric_after": metric_after, "vetoed": vetoed,
                "downweighted": len(downweighted)}

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

    def _sentry_manage_position(self, portfolio: "Portfolio", sym: str,
                                pos: "Position", px: float):
        """Mark one position to `px`, ratchet its trailing stop, and close it
        if the effective stop or take-profit is breached."""
        fx = pos.fx_rate or 1.0
        if pos.side == "BUY":
            pos.unrealized = (px - pos.entry_price) * pos.size / fx
        else:
            pos.unrealized = (pos.entry_price - px) * pos.size / fx

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
            log.info(f"SENTRY [{portfolio.id}]: {sym} hit {hit[1]} @ {px:.4f} — closing")
            self._close_position(portfolio, sym, hit[0], hit[1])
        else:
            if moved:
                trail_txt = f", trail moved to {pos.trail_stop}"
            elif pos.trail_stop is not None:
                trail_txt = f", trail at {pos.trail_stop}"
            else:
                trail_txt = ""
            log.info(f"SENTRY [{portfolio.id}]: {sym} marked @ {px:.4f}, "
                     f"unrealized {pos.unrealized:+.4f}{trail_txt}")

    async def _sentry_tick(self):
        """One Sentry pass over OPEN positions across ALL cohorts. Processes
        closes here so a concurrent Scout scan never double-touches a position
        (both pop from the same book; whoever is second finds it already gone).
        Prices are fetched ONCE for the union of symbols and shared across
        cohorts holding the same asset."""
        self.last_sentry_run = datetime.now(timezone.utc).isoformat()
        ENGINE_STATE["last_sentry_run"] = self.last_sentry_run
        # Heartbeat: proof the Sentry loop is alive, whatever else happens
        ENGINE_STATE["last_heartbeat"] = self.last_sentry_run
        # Only mark positions whose market is open — closed markets don't move.
        hours = market_hours_status()
        symbols = sorted({
            s for portfolio in self.portfolios.values() for s in portfolio.positions
            if hours.get(self.market_of(s), True)
        })
        if symbols:
            prices = await asyncio.to_thread(
                self._fetch_last_prices_sync, symbols)
            for portfolio in self.portfolios.values():
                for sym in list(portfolio.positions.keys()):
                    if sym not in prices:
                        continue
                    pos = portfolio.positions.get(sym)
                    if not pos:
                        continue  # closed elsewhere in the meantime
                    self._sentry_manage_position(portfolio, sym, pos, prices[sym])
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
        # Freeze-check the human-owned .loop anchors before anything trades.
        # A tampered anchor halts the engine — the optimizer may never edit them.
        try:
            assert_anchors_untouched()
            self._anchors_frozen = True
            self._mem("Anchors verified frozen.", "SUCCESS")
        except AnchorTamperError as e:
            self._anchors_frozen = False
            self._mem(f"ANCHOR TAMPER — halting: {e}", "FAIL")
            raise
        # Seed the dashboard loop panel from persisted .loop state before the
        # first daily cycle fires.
        _lst = self._load_loop_state()
        self.loop_best_metric = _lst.get("best_metric")
        self.loop_holdout_sharpe = _lst.get("best_holdout_sharpe")
        self.loop_open_hypotheses = _lst.get("open_hypotheses", [])
        ENGINE_STATE["loop_status"] = {
            "last_outer_run": self.last_outer_run,
            "best_metric": self.loop_best_metric,
            "holdout_sharpe": self.loop_holdout_sharpe,
            "raw_holdout_sharpe": None,
            "vetoed": None,
            "last_verdict": None,
            "downweighted_rules": [],
            "open_hypotheses": self.loop_open_hypotheses,
            "anchors_frozen": self._anchors_frozen,
            "metric_definition": ("Out-of-sample walk-forward Sharpe "
                                  "(last 20% holdout) w/ hard 2% max-DD veto"),
            "per_module": [],
        }
        self._mem(f"Engine started. Capital=${self.initial_capital} Target=${self.target_equity}")
        if len(self.portfolios) > 1 or self.primary.id != "DISTRIBUTED":
            self._mem(
                f"COHORT MODE: {len(self.portfolios)} isolated portfolios, "
                f"each ${INITIAL_CAPITAL:.0f} — "
                + " | ".join(f"{p.name} ({p.market_label()})"
                             for p in self.portfolios.values())
                + f" — shared scan/FinBERT/brain, primary={self.primary.id}",
                "SUCCESS")
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
            for p in self.portfolios.values():
                await self.store.save_engine_state(self._engine_state_row(p))
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
