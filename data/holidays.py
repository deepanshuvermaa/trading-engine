"""
Market-holiday calendars.
=========================
Trading holidays for the exchanges the engine watches. On these dates the
respective cash market is fully closed (no session), independent of the usual
weekday/clock gate. Crypto is unaffected — it never observes a holiday.

Dates are stored as ISO strings ("YYYY-MM-DD") keyed by year so the lookups
stay obvious and are trivial to extend. Weekend-only exchange closures are NOT
listed here; the weekday check in market_hours_status() already handles those.
"""

from __future__ import annotations

from datetime import date

# ── NSE (India) trading holidays ──────────────────────────────────────────
# 2026 full-day trading holidays. Muhurat/special sessions are ignored.
NSE_HOLIDAYS: dict[int, dict[str, str]] = {
    2026: {
        "2026-01-26": "Republic Day",
        "2026-03-04": "Holi",
        "2026-04-03": "Good Friday",
        "2026-08-15": "Independence Day",
        "2026-10-02": "Gandhi Jayanti",
        "2026-11-08": "Diwali (Laxmi Pujan)",
        "2026-12-25": "Christmas",
    },
}

# ── NYSE (US) trading holidays ────────────────────────────────────────────
# 2026 observed holidays per the NYSE calendar.
NYSE_HOLIDAYS: dict[int, dict[str, str]] = {
    2026: {
        "2026-01-01": "New Year's Day",
        "2026-01-19": "Martin Luther King Jr. Day",
        "2026-02-16": "Washington's Birthday",
        "2026-04-03": "Good Friday",
        "2026-05-25": "Memorial Day",
        "2026-06-19": "Juneteenth National Independence Day",
        "2026-07-03": "Independence Day (observed)",
        "2026-09-07": "Labor Day",
        "2026-11-26": "Thanksgiving Day",
        "2026-12-25": "Christmas Day",
    },
}


def is_nse_holiday(d: date) -> bool:
    """True if the NSE cash market is closed for a holiday on `d`.
    Unknown year -> False (update me for future years)."""
    return d.isoformat() in NSE_HOLIDAYS.get(d.year, {})


def is_nyse_holiday(d: date) -> bool:
    """True if the NYSE is closed for a holiday on `d`.
    Unknown year -> False (update me for future years)."""
    return d.isoformat() in NYSE_HOLIDAYS.get(d.year, {})


def holiday_name_nse(d: date) -> str | None:
    """Human-readable NSE holiday label for `d`, or None."""
    return NSE_HOLIDAYS.get(d.year, {}).get(d.isoformat())


def holiday_name_nyse(d: date) -> str | None:
    """Human-readable NYSE holiday label for `d`, or None."""
    return NYSE_HOLIDAYS.get(d.year, {}).get(d.isoformat())
