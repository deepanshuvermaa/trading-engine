"""Weekly research note — equity-research / IB-grade audit report.

Builds a self-contained, print-ready HTML report (Ctrl+P -> PDF) and a CSV
trade blotter from the durable audit trail:
  - reports/audit/attribution.jsonl  (closed trades, full attribution + rule cites)
  - reports/audit/audit_*.jsonl       (decisions taken / rejected / skipped)
  - knowledge/rule_stats.json         (investor-rule win/loss)
  - .loop/state.json                  (held-out Sharpe, best config)
  - ENGINE_STATE (live)               (cohorts, cost drag, equity curve, config)

Reads defensively: any missing source degrades to an empty section, never
crashes. Works with or without Postgres (files are the durable fallback).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
AUDIT_DIR = Path("./reports/audit")
RULE_STATS = Path("./knowledge/rule_stats.json")
LOOP_STATE = Path("./.loop/state.json")
LESSONS = Path("./.loop/lessons.md")


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ist(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
    except Exception:
        return str(iso)


def _within(iso: str | None, cutoff: datetime) -> bool:
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= cutoff
    except Exception:
        return False


def _collect(days: int) -> dict:
    """Gather all report data for the trailing `days` window."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Closed trades w/ attribution
    attrib = [r for r in _load_jsonl(AUDIT_DIR / "attribution.jsonl")
              if _within(r.get("closed_at"), cutoff)]

    # Decisions (opens, rejects, skips) from all audit_*.jsonl in window
    decisions = []
    for f in sorted(AUDIT_DIR.glob("audit_*.jsonl")):
        for r in _load_jsonl(f):
            if _within(r.get("timestamp"), cutoff):
                decisions.append(r)

    rule_stats = _load_json(RULE_STATS)
    loop_state = _load_json(LOOP_STATE)
    lessons = LESSONS.read_text(encoding="utf-8") if LESSONS.exists() else ""

    return {
        "cutoff": cutoff,
        "attrib": attrib,
        "decisions": decisions,
        "rule_stats": rule_stats,
        "loop_state": loop_state,
        "lessons": lessons,
    }


def _metrics(trades: list[dict]) -> dict:
    pnls = [t.get("pnl", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    net = sum(pnls)
    return {
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(pnls) * 100) if pnls else 0.0,
        "net_pnl": net,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (gross_win and float("inf") or 0.0),
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
    }


def _esc(s) -> str:
    return (str("" if s is None else s)
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _equity_svg(curve: list[dict], w=680, h=180) -> str:
    pts = [c.get("equity") for c in curve if isinstance(c.get("equity"), (int, float))]
    if len(pts) < 2:
        return '<p class="muted">No equity curve recorded in this period.</p>'
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1
    step = w / (len(pts) - 1)
    coords = " ".join(
        f"{i*step:.1f},{h - (p-lo)/rng*(h-20) - 10:.1f}" for i, p in enumerate(pts)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" '
        f'style="border:1px solid #b8a97e;background:#f7f1e1">'
        f'<polyline points="{coords}" fill="none" stroke="#2b2416" stroke-width="1.5"/>'
        f'<text x="4" y="14" font-size="10" fill="#6a5f42">${hi:,.2f}</text>'
        f'<text x="4" y="{h-4}" font-size="10" fill="#6a5f42">${lo:,.2f}</text>'
        f'</svg>'
    )


def build_report(engine_state: dict, days: int = 7, cohort: str = "all") -> str:
    """Return a self-contained print-ready HTML research note."""
    d = _collect(days)
    trades = d["attrib"]
    if cohort and cohort != "all":
        trades = [t for t in trades if (t.get("cohort") or "").upper() == cohort.upper()]

    m = _metrics(trades)
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    frm = d["cutoff"].astimezone(IST).strftime("%Y-%m-%d")
    to = datetime.now(IST).strftime("%Y-%m-%d")

    cohorts = engine_state.get("cohorts", []) or []
    cost_drag = engine_state.get("cost_drag_total", 0.0)
    loop = engine_state.get("loop_status", {}) or {}
    holdout = loop.get("holdout_sharpe", d["loop_state"].get("best_holdout_sharpe"))
    equity = engine_state.get("equity", engine_state.get("capital", 0.0))
    capital = engine_state.get("capital", 0.0)
    peak = engine_state.get("peak_equity", capital)
    dd = engine_state.get("drawdown_pct", 0.0)

    # Verdict
    if m["n"] == 0:
        verdict = ("No positions were closed in this period. The engine either "
                   "found no setups clearing the cost + rule + reward:risk gates, "
                   "or open trades have not yet resolved. Capital preserved.")
    else:
        tone = "profitable" if m["net_pnl"] > 0 else "unprofitable"
        verdict = (f"Over the period the book closed {m['n']} trade(s), "
                   f"{m['wins']} winning and {m['losses']} losing "
                   f"({m['win_rate']:.0f}% hit rate), netting "
                   f"${m['net_pnl']:+.2f} after costs — {tone}. "
                   f"Profit factor {m['profit_factor']:.2f}. "
                   f"Held-out Sharpe {'n/a' if holdout is None else f'{holdout:+.2f}'} "
                   f"is the un-gameable verdict: only a sustained positive value "
                   f"on unseen data justifies real capital.")

    # ---- Trade blotter rows ----
    blotter = ""
    for t in sorted(trades, key=lambda x: x.get("closed_at") or "", reverse=True):
        cites = ", ".join(t.get("rule_citations", [])[:3]) if t.get("rule_citations") else "-"
        hold = "-"
        try:
            o = datetime.fromisoformat((t.get("opened_at") or "").replace("Z", "+00:00"))
            c = datetime.fromisoformat((t.get("closed_at") or "").replace("Z", "+00:00"))
            hold = f"{(c-o).total_seconds()/3600:.1f}h"
        except Exception:
            pass
        pnl = t.get("pnl", 0.0)
        cls = "pos" if pnl > 0 else "neg"
        blotter += (
            f"<tr><td>{_ist(t.get('closed_at'))}</td><td>{_esc(t.get('symbol'))}</td>"
            f"<td>{_esc(t.get('side'))}</td><td>{t.get('entry_price',0):.4f}</td>"
            f"<td>{t.get('exit_price',0):.4f}</td><td>{t.get('size',0):.6f}</td>"
            f"<td class='{cls}'>${pnl:+.4f}</td><td>{_esc(t.get('exit_reason'))}</td>"
            f"<td>{hold}</td><td>{_esc(t.get('module'))}</td><td class='cite'>{_esc(cites)}</td></tr>"
        )
    if not blotter:
        blotter = '<tr><td colspan="11" class="muted">No closed trades in this period.</td></tr>'

    # ---- Cohort scoreboard ----
    coh_rows = ""
    for c in cohorts:
        ret = c.get("return_pct", 0.0)
        coh_rows += (
            f"<tr><td>{_esc(c.get('name'))}</td><td>{_esc(c.get('market_filter'))}</td>"
            f"<td>${c.get('equity',0):,.2f}</td><td class='{'pos' if ret>=0 else 'neg'}'>{ret:+.2f}%</td>"
            f"<td>{c.get('closed_trades',0)}</td><td>{c.get('drawdown',c.get('drawdown_pct',0)):.2f}%</td>"
            f"<td>${c.get('cost_drag',0):.4f}</td></tr>"
        )
    if not coh_rows:
        coh_rows = '<tr><td colspan="7" class="muted">Single-portfolio mode (no cohorts).</td></tr>'

    # ---- Decisions taken (opens) ----
    opens = [x for x in d["decisions"] if x.get("type") == "DECISION_OPEN"][:15]
    dec_rows = ""
    for x in opens:
        dec_rows += (f"<li><b>{_esc(x.get('symbol'))}</b> {_esc(x.get('direction'))} — "
                     f"{_esc(x.get('reasoning',''))[:240]}</li>")
    if not dec_rows:
        dec_rows = '<li class="muted">No opening decisions recorded in this period.</li>'

    # ---- What we missed / rejected ----
    rejects = [x for x in d["decisions"]
               if x.get("type") in ("DECISION_REJECT", "DECISION_SKIP")][:25]
    rej_rows = ""
    for x in rejects:
        rej_rows += (f"<li><b>{_esc(x.get('symbol'))}</b> — "
                     f"{_esc(x.get('reasoning', x.get('decision','')))[:200]}</li>")
    if not rej_rows:
        rej_rows = '<li class="muted">No rejected setups logged in this period.</li>'

    # ---- Rule performance ----
    rs = d["rule_stats"]
    ranked = sorted(
        [(k, v) for k, v in rs.items() if (v.get("wins", 0) + v.get("losses", 0)) > 0],
        key=lambda kv: (kv[1].get("wins", 0) - kv[1].get("losses", 0)), reverse=True,
    )
    top = ranked[:8]
    bottom = ranked[-8:][::-1] if len(ranked) > 8 else []
    def _rule_rows(items):
        out = ""
        for k, v in items:
            w, l = v.get("wins", 0), v.get("losses", 0)
            acc = (w / (w + l) * 100) if (w + l) else 0
            out += (f"<tr><td>{_esc(k)}</td><td>{w}</td><td>{l}</td>"
                    f"<td>{acc:.0f}%</td><td>{v.get('weight',1.0):.3f}</td></tr>")
        return out or '<tr><td colspan="5" class="muted">No rule track record yet.</td></tr>'

    # ---- Cost analysis ----
    gross = m["gross_win"] + m["gross_loss"]
    cost_pct = (cost_drag / gross * 100) if gross > 0 else 0.0

    lessons_html = _esc(d["lessons"])[:4000].replace("\n", "<br>") or \
        '<span class="muted">No lessons recorded yet.</span>'

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Dalal Street Journal — Weekly Research Note</title>
<style>
@page {{ margin: 18mm; }}
body {{ font-family: Georgia,'Times New Roman',serif; color:#2b2416; background:#f4ecd8;
  max-width:820px; margin:0 auto; padding:24px; line-height:1.5; }}
h1 {{ font-size:26px; border-bottom:3px double #2b2416; padding-bottom:8px; margin:0 0 4px; }}
h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid #b8a97e;
  padding-bottom:4px; margin:26px 0 10px; }}
.sub {{ color:#6a5f42; font-style:italic; font-size:13px; margin-bottom:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; margin:8px 0; }}
th,td {{ text-align:left; padding:4px 6px; border-bottom:1px solid #d8cba5; }}
th {{ background:#e8dcbb; text-transform:uppercase; font-size:10px; letter-spacing:.5px; }}
.pos {{ color:#1a6b32; font-weight:bold; }} .neg {{ color:#9b2226; font-weight:bold; }}
.muted {{ color:#9a8c68; font-style:italic; }}
.cite {{ font-size:10px; color:#6a5f42; }}
.kpi {{ display:flex; flex-wrap:wrap; gap:14px; margin:10px 0; }}
.kpi div {{ background:#efe6cb; border:1px solid #cdbd90; padding:8px 12px; min-width:120px; }}
.kpi b {{ display:block; font-size:19px; }} .kpi span {{ font-size:10px; text-transform:uppercase; color:#6a5f42; }}
ul {{ font-size:12px; padding-left:18px; }} li {{ margin:5px 0; }}
.foot {{ margin-top:30px; border-top:1px solid #b8a97e; padding-top:10px; font-size:11px; color:#6a5f42; }}
@media print {{ body {{ background:#fff; }} .noprint {{ display:none; }} }}
.noprint {{ text-align:right; margin-bottom:10px; }}
button {{ font-family:Georgia,serif; padding:6px 14px; cursor:pointer; }}
</style></head><body>
<div class="noprint"><button onclick="window.print()">Print / Save as PDF</button></div>
<h1>THE DALAL STREET JOURNAL</h1>
<div class="sub">Weekly Research Note &middot; {frm} to {to} &middot; Cohort: {_esc(cohort)} &middot; Generated {now_ist}</div>

<h2>Executive Summary</h2>
<div class="kpi">
  <div><span>Capital</span><b>${capital:,.2f}</b></div>
  <div><span>Equity</span><b>${equity:,.2f}</b></div>
  <div><span>Net P&amp;L</span><b class="{'pos' if m['net_pnl']>=0 else 'neg'}">${m['net_pnl']:+.2f}</b></div>
  <div><span>Held-out Sharpe</span><b>{'n/a' if holdout is None else f'{holdout:+.2f}'}</b></div>
  <div><span>Max Drawdown</span><b>{dd:.2f}%</b></div>
  <div><span>Win Rate</span><b>{m['win_rate']:.0f}%</b></div>
  <div><span>Profit Factor</span><b>{m['profit_factor']:.2f}</b></div>
  <div><span>Trades</span><b>{m['n']}</b></div>
  <div><span>Cost Paid</span><b>${cost_drag:.4f}</b></div>
</div>
<p>{verdict}</p>

<h2>Per-Cohort Scoreboard — Where the Edge Lived</h2>
<table><tr><th>Cohort</th><th>Markets</th><th>Equity</th><th>Return</th><th>Trades</th><th>Max DD</th><th>Cost Drag</th></tr>{coh_rows}</table>

<h2>Performance — Equity Curve</h2>
{_equity_svg(engine_state.get('equity_curve', []))}

<h2>Trade Blotter</h2>
<table><tr><th>Closed (IST)</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Size</th><th>P&amp;L (USD)</th><th>Reason</th><th>Held</th><th>Module</th><th>Rules Cited</th></tr>{blotter}</table>

<h2>Decisions Taken</h2>
<ul>{dec_rows}</ul>

<h2>What We Missed / Rejected — Opportunities Not Taken</h2>
<ul>{rej_rows}</ul>

<h2>Rule Performance — The Legends' Scorecard</h2>
<b style="font-size:12px">Top performers</b>
<table><tr><th>Rule</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>Weight</th></tr>{_rule_rows(top)}</table>
{'<b style="font-size:12px">Weakest (down-weighted)</b><table><tr><th>Rule</th><th>Wins</th><th>Losses</th><th>Accuracy</th><th>Weight</th></tr>' + _rule_rows(bottom) + '</table>' if bottom else ''}

<h2>Cost Analysis</h2>
<p>Total transaction cost (fees + slippage) this period: <b>${cost_drag:.4f}</b>.
Gross P&amp;L moved through <b>${gross:.4f}</b>, so cost consumed <b>{cost_pct:.1f}%</b> of gross activity.
Per-market: US equities ~0.06% round-trip, crypto ~0.30%, NSE dominated by the flat brokerage
(unviable at small size — correctly rejected).</p>

<h2>Lessons Learned</h2>
<p style="font-size:12px">{lessons_html}</p>

<div class="foot">
Paper trading — no real capital at risk. Signals are deterministic (no LLM in the trading loop).
Optimization anchors are frozen and hash-verified; the held-out Sharpe metric cannot be optimized against.
All prices sourced from public market data (yfinance / NSE). Every trade above is verifiable against
public records. This note is generated automatically from the engine's durable audit trail.
</div>
</body></html>"""


def build_csv(days: int = 7, cohort: str = "all") -> str:
    """Return the closed-trade blotter as CSV."""
    d = _collect(days)
    trades = d["attrib"]
    if cohort and cohort != "all":
        trades = [t for t in trades if (t.get("cohort") or "").upper() == cohort.upper()]

    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["closed_at_ist", "symbol", "side", "entry_price", "exit_price",
                 "size", "pnl_usd", "ret_pct", "exit_reason", "module",
                 "regime", "rules_cited"])
    for t in sorted(trades, key=lambda x: x.get("closed_at") or "", reverse=True):
        wr.writerow([
            _ist(t.get("closed_at")), t.get("symbol"), t.get("side"),
            t.get("entry_price"), t.get("exit_price"), t.get("size"),
            round(t.get("pnl", 0.0), 4), round(t.get("ret_pct", 0.0), 4),
            t.get("exit_reason"), t.get("module"), t.get("regime"),
            "; ".join(t.get("rule_citations", []) or []),
        ])
    return buf.getvalue()
