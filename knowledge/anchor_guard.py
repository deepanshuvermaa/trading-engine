"""Anchor guard — the optimizer must never edit the frozen human-owned anchors.

The three files under `.loop/anchors/` (metrics.md, holdout.md, veto.md) encode
the confirmed metric, the held-out scheme, and the never-weaken invariants. They
are human-owned ground truth. This module hashes them and compares against a
stored baseline (`.loop/anchors/.hashes.json`); if any anchor changed it raises
`AnchorTamperError`, which halts the engine.

Called at engine startup (autonomous.py) and before every daily learning loop.
On the very first run the baseline is recorded (there is nothing to compare to
yet) and the guard passes; every run after must match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utils.logger import get_logger

log = get_logger("knowledge.anchor_guard")

# Repo root = parent of the knowledge/ package directory.
_ROOT = Path(__file__).resolve().parent.parent
ANCHORS_DIR = _ROOT / ".loop" / "anchors"
HASHES_PATH = ANCHORS_DIR / ".hashes.json"

# The exact files that are frozen. Order-independent (keyed by filename).
ANCHOR_FILES = ("metrics.md", "holdout.md", "veto.md")


class AnchorTamperError(RuntimeError):
    """Raised when a frozen anchor file no longer matches its recorded hash."""


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes() -> dict[str, str]:
    """sha256 of each anchor file. Missing files hash to the sentinel MISSING."""
    out: dict[str, str] = {}
    for name in ANCHOR_FILES:
        p = ANCHORS_DIR / name
        out[name] = _hash_file(p) if p.exists() else "MISSING"
    return out


def _load_baseline() -> dict[str, str] | None:
    if not HASHES_PATH.exists():
        return None
    try:
        return json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"anchor_guard: could not read baseline hashes ({e}); "
                    "treating as first run")
        return None


def _write_baseline(hashes: dict[str, str]) -> None:
    ANCHORS_DIR.mkdir(parents=True, exist_ok=True)
    HASHES_PATH.write_text(json.dumps(hashes, indent=2, sort_keys=True),
                           encoding="utf-8")


def assert_anchors_untouched(record_if_missing: bool = True) -> dict[str, str]:
    """Verify the frozen anchors are unchanged. Returns the current hashes.

    - First run (no baseline): records the baseline and passes (if
      `record_if_missing`), else raises.
    - Any anchor file missing -> raises (a frozen anchor may not be deleted).
    - Any hash mismatch -> raises AnchorTamperError (the engine must halt).
    """
    missing = [n for n in ANCHOR_FILES if not (ANCHORS_DIR / n).exists()]
    if missing:
        raise AnchorTamperError(
            f"Frozen anchor file(s) missing: {', '.join(missing)} "
            f"(expected under {ANCHORS_DIR})")

    cur = current_hashes()
    baseline = _load_baseline()

    if baseline is None:
        if not record_if_missing:
            raise AnchorTamperError(
                "No anchor baseline recorded and record_if_missing=False")
        _write_baseline(cur)
        log.info("Anchors verified frozen (baseline recorded).")
        return cur

    changed = [n for n in ANCHOR_FILES if baseline.get(n) != cur.get(n)]
    if changed:
        raise AnchorTamperError(
            "FROZEN ANCHOR CHANGED — the optimizer must never edit anchors. "
            f"Mismatched: {', '.join(changed)}. Files under {ANCHORS_DIR} were "
            "modified since the baseline in .hashes.json. Restore them or, if the "
            "change is human-intended, delete .hashes.json to re-baseline.")

    log.info("Anchors verified frozen.")
    return cur


if __name__ == "__main__":
    assert_anchors_untouched()
    print("Anchors verified frozen.")
