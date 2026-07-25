"""
Decision Audit Log — analyst-readable reasoning for every action.

Every trade decision records:
- WHAT triggered it (which indicators, what values)
- WHY this direction (the scoring logic chain)
- WHAT the market context was (trend, regime, S/R levels)
- WHAT risk parameters were applied
- OUTCOME and LESSON when trade closes

This log is designed so a human analyst can:
1. Read it like a trade journal
2. Identify recurring mistakes
3. Suggest strategy improvements
4. Verify every claim against public market data
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db.store import get_store


class DecisionAudit:
    """Analyst-grade decision logging."""

    def __init__(self, log_dir: str = "./reports/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        self._file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log_scan(self, symbol: str, data: dict) -> dict:
        """Log a market scan with full indicator snapshot."""
        entry = {
            "type": "SCAN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "market_snapshot": {
                "price": data.get("current_price"),
                "rsi": data.get("rsi"),
                "adx": data.get("adx"),
                "macd_hist": data.get("macd_hist", 0),
                "volume_ratio": data.get("volume_ratio"),
                "atr": data.get("atr"),
                "pct_5d": data.get("pct_5d"),
                "pct_20d": data.get("pct_20d"),
                "above_sma200": data.get("above_sma200"),
                "golden_cross": data.get("golden_cross"),
                "support": data.get("support", []),
                "resistance": data.get("resistance", []),
            },
            "score": data.get("score", 0),
            "direction": data.get("direction"),
            "confidence": data.get("confidence"),
            "risk_reward": data.get("risk_reward"),
            "factors": data.get("reasons", []),
        }
        self._persist(entry)
        return entry

    def log_trade_decision(
        self,
        action: str,  # OPEN, SKIP, REJECT
        symbol: str,
        setup: dict,
        portfolio_context: dict,
        reasoning: str,
    ) -> dict:
        """Log a trade decision with full context."""
        entry = {
            "type": f"DECISION_{action}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": setup.get("direction"),
            "decision": action,

            "signal_analysis": {
                "score": setup.get("score"),
                "confidence": setup.get("confidence"),
                "risk_reward": setup.get("risk_reward"),
                "module": setup.get("module"),
                "entry_price": setup.get("current_price"),
                "stop_loss": setup.get("stop_loss"),
                "take_profit": setup.get("take_profit"),
            },

            "indicator_evidence": {
                "rsi": {"value": setup.get("rsi"), "interpretation": self._interpret_rsi(setup.get("rsi", 50))},
                "adx": {"value": setup.get("adx"), "interpretation": self._interpret_adx(setup.get("adx", 0))},
                "volume": {"ratio": setup.get("volume_ratio"), "interpretation": self._interpret_volume(setup.get("volume_ratio", 1))},
                "trend": {"sma200": setup.get("above_sma200"), "golden_cross": setup.get("golden_cross")},
                "momentum_5d": setup.get("pct_5d"),
                "momentum_20d": setup.get("pct_20d"),
            },

            "scoring_breakdown": setup.get("reasons", []),

            "portfolio_context": {
                "equity_before": portfolio_context.get("equity"),
                "drawdown_pct": portfolio_context.get("drawdown_pct"),
                "open_positions": portfolio_context.get("open_positions"),
                "position_size_usd": portfolio_context.get("position_size"),
                "risk_usd": portfolio_context.get("risk_amount"),
                "risk_pct_of_equity": portfolio_context.get("risk_pct"),
            },

            "reasoning": reasoning,
        }
        self._persist(entry)
        return entry

    def log_trade_outcome(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        duration_hours: float,
        entry_reasons: list[str],
        market_at_exit: dict,
    ) -> dict:
        """Log trade outcome with lesson learned."""

        # Auto-generate lesson
        lesson = self._generate_lesson(
            side, entry_price, exit_price, pnl, exit_reason, entry_reasons, market_at_exit
        )

        entry = {
            "type": "OUTCOME",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "trade": {
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_usd": round(pnl, 4),
                "pnl_pct": round(pnl / (entry_price * 0.001) * 100, 2) if entry_price else 0,
                "exit_reason": exit_reason,
                "duration_hours": round(duration_hours, 1),
            },
            "entry_factors": entry_reasons,
            "exit_market_state": market_at_exit,
            "lesson": lesson,
            "category": "WIN" if pnl > 0 else "LOSS",
        }
        self._persist(entry)
        return entry

    def log_rule_citations(
        self,
        symbol: str,
        action: str,  # OPEN, VETO, SKIP
        citations: list[str],
        failures: list[str] | None = None,
    ) -> dict:
        """Log investor-rule citations backing a decision (knowledge/brain.py).

        Every citation carries: rule id + source investor + exact threshold vs
        the actual observed value — the audit trail for the rule layer.
        """
        entry = {
            "type": "RULE_CITATIONS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action,
            "rules_evaluated": len(citations),
            "citations": citations,
            "failed_citations": failures or [],
        }
        self._persist(entry)
        return entry

    def log_system_event(self, event: str, details: dict) -> dict:
        entry = {
            "type": "SYSTEM",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details,
        }
        self._persist(entry)
        return entry

    def get_lessons(self, category: str | None = None) -> list[dict]:
        """Get all lessons learned, optionally filtered."""
        outcomes = [e for e in self.entries if e["type"] == "OUTCOME"]
        if category:
            outcomes = [e for e in outcomes if e.get("category") == category]
        return [{"symbol": e["symbol"], "lesson": e["lesson"], "pnl": e["trade"]["pnl_usd"]} for e in outcomes]

    def get_failure_patterns(self) -> dict[str, int]:
        """Analyze recurring failure patterns."""
        patterns = {}
        losses = [e for e in self.entries if e["type"] == "OUTCOME" and e["category"] == "LOSS"]
        for loss in losses:
            for factor in loss.get("entry_factors", []):
                key = factor.split("=")[0].strip() if "=" in factor else factor
                patterns[key] = patterns.get(key, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: -x[1]))

    def generate_analyst_report(self) -> str:
        """Generate a human-readable report for analyst review."""
        outcomes = [e for e in self.entries if e["type"] == "OUTCOME"]
        decisions = [e for e in self.entries if e["type"].startswith("DECISION")]
        wins = [e for e in outcomes if e["category"] == "WIN"]
        losses = [e for e in outcomes if e["category"] == "LOSS"]

        lines = []
        lines.append("=" * 70)
        lines.append("  ANALYST REVIEW REPORT")
        lines.append(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append(f"  Decisions made: {len(decisions)}")
        lines.append(f"  Trades closed:  {len(outcomes)}")
        lines.append(f"  Wins: {len(wins)}  Losses: {len(losses)}")
        total_pnl = sum(e["trade"]["pnl_usd"] for e in outcomes)
        lines.append(f"  Net P&L: ${total_pnl:+.4f}")
        lines.append("")

        # Failure patterns
        patterns = self.get_failure_patterns()
        if patterns:
            lines.append("RECURRING FAILURE PATTERNS (factors present in losing trades)")
            for pattern, count in list(patterns.items())[:10]:
                lines.append(f"  [{count}x] {pattern}")
            lines.append("")

        # Lessons learned
        lines.append("LESSONS LEARNED")
        for outcome in outcomes[-20:]:  # Last 20
            symbol = outcome["symbol"]
            pnl = outcome["trade"]["pnl_usd"]
            cat = outcome["category"]
            lesson = outcome.get("lesson", "")
            lines.append(f"  [{cat}] {symbol} ${pnl:+.4f}: {lesson}")
        lines.append("")

        # Trade-by-trade detail
        lines.append("DETAILED TRADE LOG")
        lines.append("-" * 70)
        for outcome in outcomes:
            t = outcome["trade"]
            lines.append(f"  {outcome['symbol']} | {t['side']} | Entry ${t['entry_price']} -> Exit ${t['exit_price']}")
            lines.append(f"  P&L: ${t['pnl_usd']:+.4f} | Reason: {t['exit_reason']} | Duration: {t['duration_hours']}h")
            lines.append(f"  Entry factors:")
            for f in outcome.get("entry_factors", []):
                lines.append(f"    - {f}")
            lines.append(f"  Lesson: {outcome.get('lesson', '')}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("  END OF REPORT")
        lines.append("  This report is designed for human analyst review.")
        lines.append("  All prices verifiable against public market data.")
        lines.append("=" * 70)

        report = "\n".join(lines)

        # Save
        report_path = self.log_dir / f"analyst_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        report_path.write_text(report, encoding="utf-8")

        return report

    def _generate_lesson(
        self, side, entry, exit_p, pnl, reason, entry_reasons, market_at_exit
    ) -> str:
        """Auto-generate a lesson from a trade outcome."""
        if pnl > 0:
            if reason == "TAKE_PROFIT":
                return f"Signal confirmed. Entry factors held: {', '.join(entry_reasons[:2])}. Full target reached."
            else:
                return f"Partial win. Closed at {reason}. Entry thesis was correct but timing could improve."
        else:
            if reason == "STOP_LOSS":
                exit_rsi = market_at_exit.get("rsi", 50)
                exit_adx = market_at_exit.get("adx", 0)

                if side == "BUY" and exit_rsi > 70:
                    return f"Bought into overbought (RSI={exit_rsi:.0f} at exit). Entry signal may have been late."
                elif side == "SELL" and exit_rsi < 30:
                    return f"Sold into oversold (RSI={exit_rsi:.0f} at exit). Counter-trend trade failed."
                elif exit_adx < 20:
                    return f"Market was ranging (ADX={exit_adx:.0f}). Trend-based entry failed in sideways market."
                else:
                    return f"Stop loss hit. Review if SL was too tight (2x ATR may need widening for this asset's volatility)."
            else:
                return f"Closed: {reason}. Review entry criteria: {', '.join(entry_reasons[:2])}"

    def _interpret_rsi(self, rsi):
        if rsi < 30: return "OVERSOLD - potential bounce"
        if rsi < 40: return "Approaching oversold"
        if rsi > 70: return "OVERBOUGHT - potential reversal"
        if rsi > 60: return "Approaching overbought"
        return "Neutral zone"

    def _interpret_adx(self, adx):
        if adx > 40: return "STRONG trend in place"
        if adx > 25: return "Trending market"
        return "Ranging/choppy - trend signals less reliable"

    def _interpret_volume(self, vr):
        if vr > 2.0: return "MAJOR volume spike - institutional activity likely"
        if vr > 1.5: return "Above-average volume - confirms move"
        if vr < 0.5: return "LOW volume - weak conviction"
        return "Normal volume"

    def _persist(self, entry: dict):
        self.entries.append(entry)
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        # Write-through to Postgres when configured (JSONL stays the local
        # fallback/backup; a DB failure never blocks the trading loop).
        store = get_store()
        if store.enabled:
            store.fire(store.log_audit_event(
                entry.get("type", "UNKNOWN"), entry.get("symbol"), entry))
