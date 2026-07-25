"""
Macro Intelligence — the news desk of the trading engine.

Aggregates three signal sources into per-market sentiment scores:
1. Curated RSS feeds (macro.feeds) — crypto / india / us / geopolitical / energy
2. FinBERT sentiment scoring (macro.sentiment) — lazy-loaded, neutral fallback
3. GDELT tone timelines (data.ingestion.macro.GDELTProvider) — global news tone

Output snapshot:
{
    "sentiments": {"crypto": -1..+1, "india": -1..+1, "us": -1..+1,
                   "geopolitical_risk": 0..1},
    "headlines": [top 10 most impactful: {title, source, sentiment, link,
                  published, market}],
    "gdelt_tone": {"crypto": t, "india": t, "us": t},
    "updated_at": iso timestamp,
}

Deterministic aggregation — no LLM anywhere in this path.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from macro.feeds import fetch_all_feeds
from macro.sentiment import score_news_batch
from data.ingestion.macro import GDELTProvider
from utils.logger import get_logger

log = get_logger("macro.intelligence")

NEUTRAL_SENTIMENT = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

# Feed category -> market bucket
CATEGORY_MARKET = {
    "crypto": "crypto",
    "india_macro": "india",
    "us_macro": "us",
    "geopolitical": "geopolitical",
    "energy": "geopolitical",
}

# GDELT tone-timeline queries per market
GDELT_QUERIES = {
    "crypto": "(bitcoin OR crypto)",
    "india": "stock market india",
    "us": "federal reserve",
}

# Blend weights when both FinBERT headlines and GDELT tone are available
FINBERT_WEIGHT = 0.7
GDELT_WEIGHT = 0.3

# ── Deterministic risk-factor tagging (keyword rules, no LLM) ──────
# (keywords, risk-factor label, markets affected). Single words match
# whole tokens; phrases (with a space) match as substrings. Empty
# markets tuple means "the article's own market".
RISK_RULES: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = [
    (("rate", "rates", "inflation", "fed", "fomc", "treasury", "yield",
      "yields", "cpi", "interest rate", "rate cut", "rate hike"),
     "Rate sensitivity: affects US equities + crypto liquidity",
     ("us", "crypto")),
    (("war", "sanction", "sanctions", "conflict", "missile", "invasion",
      "military", "airstrike", "ceasefire", "troops", "nuclear"),
     "Geopolitical risk: safe-haven flows, energy spike risk",
     ("crypto", "india", "us")),
    (("regulation", "regulatory", "regulator", "sec", "rbi", "sebi", "ban",
      "banned", "crackdown", "lawsuit", "antitrust", "probe", "fine"),
     "Regulatory risk: compliance-driven selloffs", ()),
    (("earnings", "results", "guidance", "quarterly", "revenue", "profit",
      "outlook", "forecast"),
     "Earnings catalyst: single-stock volatility", ()),
    (("hack", "hacked", "breach", "exploit", "stolen", "phishing",
      "ransomware"),
     "Security event: crypto confidence risk", ("crypto",)),
    (("oil", "crude", "opec", "gasoline", "lng", "barrel", "energy prices"),
     "Energy shock: input-cost pressure on equities", ("india", "us")),
]

_TRADEABLE_MARKETS = ("crypto", "india", "us")


def derive_risk_factors(
    title: str, summary: str, market: str
) -> tuple[list[str], list[str]]:
    """Keyword-rule tagging: (risk_factors, markets_affected)."""
    text = f"{title} {summary}".lower()
    tokens = set(re.findall(r"[a-z]+", text))
    factors: list[str] = []
    affected: set[str] = set()
    own = {market} if market in _TRADEABLE_MARKETS else set()
    for keywords, label, markets in RISK_RULES:
        hit = any(
            (k in text) if " " in k else (k in tokens)
            for k in keywords
        )
        if hit:
            factors.append(label)
            affected.update(markets or own)
    affected.update(own)
    if not affected:
        # Untagged world news: assume broad spillover
        affected = set(_TRADEABLE_MARKETS)
    order = {m: i for i, m in enumerate(_TRADEABLE_MARKETS)}
    return factors, sorted(affected, key=lambda m: order.get(m, 99))


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class MacroIntelligence:
    """Fetch + score + aggregate news into market sentiment. Cached snapshot."""

    def __init__(self, cache_minutes: int = 30):
        self.cache_minutes = cache_minutes
        self.gdelt = GDELTProvider()
        self._snapshot: dict[str, Any] | None = None
        self._fetched_at: datetime | None = None
        # Per-symbol company news cache: symbol -> (fetched_at, result)
        self._company_cache: dict[str, tuple[datetime, dict | None]] = {}

    # ── Public API ──────────────────────────────────────────────

    async def get_snapshot(
        self, force: bool = False, skip_finbert: bool = False
    ) -> dict[str, Any]:
        """Return the aggregated macro snapshot, cached for `cache_minutes`."""
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._snapshot is not None
            and self._fetched_at is not None
            and now - self._fetched_at < timedelta(minutes=self.cache_minutes)
        ):
            return self._snapshot

        articles = await self._fetch_articles()
        articles = await self._score_articles(articles, skip_finbert=skip_finbert)
        gdelt_tone = await self._fetch_gdelt_tones()

        sentiments = self._aggregate(articles, gdelt_tone)
        headlines = self._top_headlines(articles, n=10)

        self._snapshot = {
            "sentiments": sentiments,
            "headlines": headlines,
            "all_articles": self._all_articles(articles, cap=100),
            "gdelt_tone": gdelt_tone,
            "article_count": len(articles),
            "updated_at": now.isoformat(),
        }
        self._fetched_at = now
        log.info(
            f"Macro snapshot: {len(articles)} articles | "
            f"crypto {sentiments['crypto']:+.2f} india {sentiments['india']:+.2f} "
            f"us {sentiments['us']:+.2f} geo-risk {sentiments['geopolitical_risk']:.2f}"
        )
        return self._snapshot

    async def company_news(
        self, symbol: str, market: str, skip_finbert: bool = False
    ) -> dict[str, Any] | None:
        """Up to 5 recent yfinance headlines for one symbol, FinBERT-scored.

        Returns {"direction": -1..+1, "count": n, "headlines": [...]} or None.
        yfinance news is flaky — every failure path returns None quietly.
        """
        now = datetime.now(timezone.utc)
        cached = self._company_cache.get(symbol)
        if cached and now - cached[0] < timedelta(minutes=self.cache_minutes):
            return cached[1]

        result: dict[str, Any] | None = None
        try:
            ticker = self._yf_ticker(symbol, market)
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, self._fetch_yf_news, ticker)
            items = self._parse_yf_news(raw)[:5]
            if items:
                if skip_finbert:
                    for it in items:
                        it["sentiment"] = dict(NEUTRAL_SENTIMENT)
                        it["direction"] = 0.0
                else:
                    try:
                        items = await loop.run_in_executor(
                            None, score_news_batch, items
                        )
                    except Exception as e:
                        log.error(f"FinBERT failed on {symbol} news: {e}")
                        for it in items:
                            it.setdefault("sentiment", dict(NEUTRAL_SENTIMENT))
                            it.setdefault("direction", 0.0)
                dirs = [it.get("direction", 0.0) for it in items]
                result = {
                    "direction": round(sum(dirs) / len(dirs), 3),
                    "count": len(items),
                    "headlines": [
                        {
                            "title": it.get("title", ""),
                            "sentiment": round(it.get("direction", 0.0), 3),
                            "link": it.get("link", ""),
                            "published": it.get("published", ""),
                        }
                        for it in items
                    ],
                }
        except Exception as e:
            log.warning(f"Company news failed for {symbol}: {e}")
            result = None

        self._company_cache[symbol] = (now, result)
        return result

    # ── Fetch layers ────────────────────────────────────────────

    async def _fetch_articles(self) -> list[dict[str, Any]]:
        """Fetch all RSS feeds and flatten with a market tag per article."""
        try:
            by_category = await fetch_all_feeds()
        except Exception as e:
            log.error(f"RSS fetch failed entirely: {e}")
            return []

        articles: list[dict[str, Any]] = []
        for category, entries in (by_category or {}).items():
            market = CATEGORY_MARKET.get(category, "geopolitical")
            for entry in entries:
                entry["market"] = market
                entry["category"] = category
                articles.append(entry)
        return articles

    async def _score_articles(
        self, articles: list[dict[str, Any]], skip_finbert: bool = False
    ) -> list[dict[str, Any]]:
        """Attach FinBERT sentiment. Neutral fallback on any failure."""
        if not articles:
            return articles
        if skip_finbert:
            for a in articles:
                a["sentiment"] = dict(NEUTRAL_SENTIMENT)
                a["direction"] = 0.0
            return articles
        try:
            loop = asyncio.get_event_loop()
            # FinBERT inference is CPU-bound — keep it off the event loop.
            return await loop.run_in_executor(None, score_news_batch, articles)
        except Exception as e:
            log.error(f"FinBERT scoring failed, falling back to neutral: {e}")
            for a in articles:
                a["sentiment"] = dict(NEUTRAL_SENTIMENT)
                a["direction"] = 0.0
            return articles

    async def _fetch_gdelt_tones(self) -> dict[str, float | None]:
        """Recent GDELT tone per market, normalised to roughly -1..+1."""
        tones: dict[str, float | None] = {}
        for market, query in GDELT_QUERIES.items():
            try:
                df = await self.gdelt.get_tone_timeline(query, timespan="7days")
                if df is not None and not df.empty and "tone" in df.columns:
                    recent = df["tone"].tail(7)
                    raw = float(recent.mean())
                    tones[market] = round(_clip(raw / 10.0), 3)
                else:
                    tones[market] = None
            except Exception as e:
                log.error(f"GDELT tone failed for {market}: {e}")
                tones[market] = None
        return tones

    # ── Aggregation ─────────────────────────────────────────────

    def _aggregate(
        self,
        articles: list[dict[str, Any]],
        gdelt_tone: dict[str, float | None],
    ) -> dict[str, float]:
        """Blend FinBERT headline sentiment with GDELT tone per market."""
        by_market: dict[str, list[dict]] = {}
        for a in articles:
            by_market.setdefault(a.get("market", "geopolitical"), []).append(a)

        sentiments: dict[str, float] = {}
        for market in ("crypto", "india", "us"):
            arts = by_market.get(market, [])
            fin = (
                sum(a.get("direction", 0.0) for a in arts) / len(arts)
                if arts
                else None
            )
            gd = gdelt_tone.get(market)
            if fin is not None and gd is not None:
                blended = FINBERT_WEIGHT * fin + GDELT_WEIGHT * gd
            elif fin is not None:
                blended = fin
            elif gd is not None:
                blended = gd
            else:
                blended = 0.0
            sentiments[market] = round(_clip(blended), 3)

        # Geopolitical risk 0..1 — mean FinBERT negative probability across
        # world-news + energy headlines. 0.33 = neutral wire, 1.0 = all bad news.
        geo = by_market.get("geopolitical", [])
        if geo:
            neg = sum(
                a.get("sentiment", NEUTRAL_SENTIMENT).get("negative", 0.33)
                for a in geo
            ) / len(geo)
            sentiments["geopolitical_risk"] = round(_clip(neg, 0.0, 1.0), 3)
        else:
            sentiments["geopolitical_risk"] = 0.0

        return sentiments

    @staticmethod
    def _top_headlines(
        articles: list[dict[str, Any]], n: int = 10
    ) -> list[dict[str, Any]]:
        """Top-n most impactful headlines, ranked by |sentiment direction|."""
        ranked = sorted(
            (a for a in articles if a.get("title")),
            key=lambda a: abs(a.get("direction", 0.0)),
            reverse=True,
        )
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "sentiment": round(a.get("direction", 0.0), 3),
                "link": a.get("link", ""),
                "published": a.get("published", ""),
                "market": a.get("market", ""),
            }
            for a in ranked[:n]
        ]

    @staticmethod
    def _all_articles(
        articles: list[dict[str, Any]], cap: int = 100
    ) -> list[dict[str, Any]]:
        """Full scored article list for the News Desk tab. Ranked by
        |sentiment| so the most impactful survive the cap, each tagged with
        deterministic risk factors + affected markets."""
        ranked = sorted(
            (a for a in articles if a.get("title")),
            key=lambda a: abs(a.get("direction", 0.0)),
            reverse=True,
        )[:cap]
        out: list[dict[str, Any]] = []
        for a in ranked:
            factors, affected = derive_risk_factors(
                a.get("title", ""), a.get("summary", "") or "",
                a.get("market", ""))
            out.append({
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "sentiment": round(a.get("direction", 0.0), 3),
                "link": a.get("link", ""),
                "published": a.get("published", ""),
                "market": a.get("market", ""),
                "category": a.get("category", ""),
                "risk_factors": factors,
                "markets_affected": affected,
            })
        return out

    # ── yfinance helpers ────────────────────────────────────────

    @staticmethod
    def _yf_ticker(symbol: str, market: str) -> str:
        """Map engine symbols to Yahoo Finance tickers."""
        if market == "crypto" or "/" in symbol:
            return symbol.split("/")[0] + "-USD"
        if market == "india":
            return symbol + ".NS"
        return symbol

    @staticmethod
    def _fetch_yf_news(ticker: str) -> list[dict]:
        import yfinance as yf

        news = yf.Ticker(ticker).news
        return list(news) if news else []

    @staticmethod
    def _parse_yf_news(raw: list[dict]) -> list[dict[str, Any]]:
        """Normalise both old and new yfinance news schemas."""
        items = []
        for it in raw or []:
            try:
                if "content" in it and isinstance(it["content"], dict):
                    # yfinance >= 0.2.5x schema
                    c = it["content"]
                    title = c.get("title", "")
                    summary = (c.get("summary") or c.get("description") or "")[:500]
                    link = (c.get("canonicalUrl") or {}).get("url", "")
                    source = (c.get("provider") or {}).get("displayName", "")
                    published = c.get("pubDate", "")
                else:
                    # legacy schema
                    title = it.get("title", "")
                    summary = ""
                    link = it.get("link", "")
                    source = it.get("publisher", "")
                    ts = it.get("providerPublishTime")
                    published = (
                        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                        if ts
                        else ""
                    )
                if title:
                    items.append(
                        {
                            "title": title,
                            "summary": summary,
                            "link": link,
                            "source": source,
                            "published": published,
                        }
                    )
            except Exception:
                continue
        return items
