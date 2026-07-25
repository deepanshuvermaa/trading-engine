"""Centralized logging setup."""

import io
import logging
import sys
from collections import deque
from datetime import datetime, timezone

# ── Ring buffer: the last 500 records from ALL loggers ──────────
# A single handler on the root logger sees every record that child
# loggers propagate (the default), regardless of which module made it.
_RING: deque = deque(maxlen=500)


class RingBufferHandler(logging.Handler):
    """Capture every log record into a shared in-memory ring buffer."""

    def emit(self, record: logging.LogRecord):
        try:
            _RING.append({
                "ts": datetime.fromtimestamp(
                    record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            })
        except Exception:
            self.handleError(record)


def _install_ring_buffer():
    root = logging.getLogger()
    if not any(isinstance(h, RingBufferHandler) for h in root.handlers):
        handler = RingBufferHandler(level=logging.NOTSET)
        root.addHandler(handler)


def _install_file_log():
    """Durable rotating log file so every action survives restarts for later read."""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        log_dir / "engine.log", maxBytes=10_000_000, backupCount=10, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)
    root.setLevel(logging.INFO)


_install_ring_buffer()
_install_file_log()


def get_recent_logs(limit: int | None = None) -> list[dict]:
    """Most recent log records, oldest first. `limit` caps the tail."""
    items = list(_RING)
    if limit is not None and limit > 0:
        items = items[-limit:]
    return items


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Force UTF-8 on Windows to avoid cp1252 encoding errors
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
