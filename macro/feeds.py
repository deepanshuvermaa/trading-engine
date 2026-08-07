"""RSS feed aggregation — curated news sources for macro intelligence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import feedparser

from utils.logger import get_logger

log = get_logger("macro.feeds")

# Curated feed sources — relevant to our 4 markets
FEED_SOURCES = {
    "india_macro": [
        ("RBI Press", "https://rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?format=rss"),
        ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("MoneyControl", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ],
    "us_macro": [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ],
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("The Block", "https://www.theblock.co/rss.xml"),
    ],
    "geopolitical": [
        ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
        ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ],
    "energy": [
        ("OilPrice", "https://oilprice.com/rss/main"),
    ],
}


async def fetch_feed(name: str, url: str) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed."""
    loop = asyncio.get_event_loop()
    try:
        parsed = await loop.run_in_executor(None, feedparser.parse, url)
        entries = []
        for entry in parsed.entries[:20]:  # Latest 20 per feed
            entries.append({
                "source": name,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:500],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        return entries
    except Exception as e:
        log.error(f"Feed error {name}: {e}")
        return []


async def fetch_all_feeds(
    categories: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all feeds, optionally filtered by category."""
    cats = categories or list(FEED_SOURCES.keys())
    results = {}

    for cat in cats:
        feeds = FEED_SOURCES.get(cat, [])
        tasks = [fetch_feed(name, url) for name, url in feeds]
        category_entries = []
        for coro in asyncio.as_completed(tasks):
            entries = await coro
            category_entries.extend(entries)
        results[cat] = category_entries
        log.info(f"Feeds [{cat}]: {len(category_entries)} articles")

    return results
