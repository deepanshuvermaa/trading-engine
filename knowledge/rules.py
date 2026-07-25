"""Investor rule registry — structured access to docs/INVESTOR_RULES_SPEC.md.

Loads knowledge/rules.yaml into pydantic Rule objects. Each rule carries:
- id + source investor (the citation anchor)
- exact threshold (human-readable) + machine params
- IMMUTABLE / LEARNABLE flag (spec Appendix + §10.4 veto list)

IMMUTABLE rules are risk-engine vetoes: the learning loop may NEVER relax them.
LEARNABLE thresholds may only be mutated via loops/experiment.py.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from db.store import get_store
from utils.logger import get_logger

log = get_logger("knowledge.rules")

RULES_YAML = Path(__file__).parent / "rules.yaml"
RULE_STATS_PATH = Path(__file__).parent / "rule_stats.json"


class Mutability(str, Enum):
    IMMUTABLE = "IMMUTABLE"
    LEARNABLE = "LEARNABLE"


class Rule(BaseModel):
    """One codified investor rule."""

    id: str
    investor: str
    name: str
    threshold: str  # exact threshold, human-readable (the citation text)
    params: dict[str, Any] = Field(default_factory=dict)
    mutability: Mutability = Mutability.LEARNABLE
    confidence: str = "MOD"  # HIGH / MOD / LOW per spec
    category: str = "general"
    sleeve: str = "all"

    @property
    def immutable(self) -> bool:
        return self.mutability == Mutability.IMMUTABLE

    def cite(self) -> str:
        """Canonical citation string: rule id + investor + threshold."""
        return f"[{self.id}|{self.investor}] {self.threshold}"


class RuleBook(BaseModel):
    """The full loaded rule set with lookup helpers."""

    version: int = 1
    source: str = ""
    rules: dict[str, Rule] = Field(default_factory=dict)

    def get(self, rule_id: str) -> Rule | None:
        return self.rules.get(rule_id)

    def all(self) -> list[Rule]:
        return list(self.rules.values())

    def immutable(self) -> list[Rule]:
        return [r for r in self.rules.values() if r.immutable]

    def learnable(self) -> list[Rule]:
        return [r for r in self.rules.values() if not r.immutable]

    def by_investor(self, investor: str) -> list[Rule]:
        return [r for r in self.rules.values() if r.investor == investor]

    def by_category(self, category: str) -> list[Rule]:
        return [r for r in self.rules.values() if r.category == category]

    def investors(self) -> list[str]:
        return sorted({r.investor for r in self.rules.values()})


def load_rules(path: str | Path = RULES_YAML) -> RuleBook:
    """Parse knowledge/rules.yaml into a RuleBook."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules: dict[str, Rule] = {}
    for item in raw.get("rules", []):
        rule = Rule.model_validate(item)
        if rule.id in rules:
            raise ValueError(f"Duplicate rule id in {path}: {rule.id}")
        rules[rule.id] = rule
    book = RuleBook(version=raw.get("version", 1), source=raw.get("source", str(path)), rules=rules)
    log.info(
        f"Loaded {len(book.rules)} investor rules "
        f"({len(book.immutable())} IMMUTABLE, {len(book.learnable())} LEARNABLE)"
    )
    return book


class RuleStats:
    """Per-rule win/loss ledger — the outer loop's evidence for down-weighting.

    A rule is scored "correct" on a closed trade when its verdict matched the
    outcome: (passed AND trade won) or (failed AND trade lost). LEARNABLE rules
    whose accuracy decays get weight < 1.0; IMMUTABLE rules are always weight
    1.0 and may never be relaxed regardless of stats.
    """

    MIN_SAMPLES = 10
    WEIGHT_FLOOR = 0.5
    WEIGHT_CAP = 1.5

    def __init__(self, path: str | Path = RULE_STATS_PATH, rulebook: RuleBook | None = None):
        self.path = Path(path)
        self.rulebook = rulebook or load_rules()
        self.stats: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.stats = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning(f"Could not parse {self.path}; starting fresh rule stats")
                self.stats = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        # Write-through to Postgres when configured — the JSON file remains
        # the local fallback/backup.
        store = get_store()
        if store.enabled:
            store.fire(store.save_rule_stats(
                json.loads(json.dumps(self.stats))))

    async def load_from_db(self) -> bool:
        """Load the ledger from Postgres (authoritative when available).

        Called on engine startup after Store.init(). Returns True if DB
        stats replaced the local file copy.
        """
        store = get_store()
        if not store.enabled:
            return False
        try:
            db_stats = await store.load_rule_stats()
        except Exception as e:
            log.warning(f"rule_stats: DB load failed, keeping file copy: {e}")
            return False
        if not db_stats:
            # First boot against an empty DB — seed it from the local file.
            if self.stats:
                await store.save_rule_stats(json.loads(json.dumps(self.stats)))
            return False
        self.stats = db_stats
        log.info(f"rule_stats: loaded {len(db_stats)} rule ledgers from Postgres")
        return True

    def record(self, rule_id: str, rule_passed: bool, trade_won: bool, pnl: float) -> None:
        """Record one closed-trade observation for a rule that was evaluated at entry."""
        s = self.stats.setdefault(
            rule_id, {"wins": 0, "losses": 0, "pnl_when_passed": 0.0, "n_passed": 0, "n_failed": 0}
        )
        correct = (rule_passed and trade_won) or (not rule_passed and not trade_won)
        if correct:
            s["wins"] += 1
        else:
            s["losses"] += 1
        if rule_passed:
            s["n_passed"] += 1
            s["pnl_when_passed"] += pnl
        else:
            s["n_failed"] += 1
        s["weight"] = self.weight(rule_id)

    def accuracy(self, rule_id: str) -> float | None:
        s = self.stats.get(rule_id)
        if not s:
            return None
        n = s["wins"] + s["losses"]
        return s["wins"] / n if n else None

    def samples(self, rule_id: str) -> int:
        s = self.stats.get(rule_id, {})
        return int(s.get("wins", 0) + s.get("losses", 0))

    def weight(self, rule_id: str) -> float:
        """Learned weight in [0.5, 1.5]. IMMUTABLE rules are pinned at 1.0 — never relaxed."""
        rule = self.rulebook.get(rule_id)
        if rule is not None and rule.immutable:
            return 1.0
        acc = self.accuracy(rule_id)
        if acc is None or self.samples(rule_id) < self.MIN_SAMPLES:
            return 1.0
        return round(max(self.WEIGHT_FLOOR, min(self.WEIGHT_CAP, 0.5 + acc)), 3)

    def underperformers(self, accuracy_below: float = 0.4) -> list[dict]:
        """LEARNABLE rules with enough samples and accuracy below threshold."""
        out = []
        for rule_id in self.stats:
            rule = self.rulebook.get(rule_id)
            if rule is None or rule.immutable:
                continue  # IMMUTABLE vetoes are never candidates for down-weighting
            acc = self.accuracy(rule_id)
            if acc is not None and self.samples(rule_id) >= self.MIN_SAMPLES and acc < accuracy_below:
                out.append(
                    {
                        "rule_id": rule_id,
                        "investor": rule.investor,
                        "category": rule.category,
                        "accuracy": round(acc, 3),
                        "samples": self.samples(rule_id),
                        "weight": self.weight(rule_id),
                    }
                )
        return sorted(out, key=lambda x: x["accuracy"])

    def snapshot(self, top_n: int = 8) -> dict:
        """Compact view for the dashboard."""
        rows = []
        for rule_id, s in self.stats.items():
            rule = self.rulebook.get(rule_id)
            rows.append(
                {
                    "id": rule_id,
                    "investor": rule.investor if rule else "?",
                    "mutability": rule.mutability.value if rule else "?",
                    "wins": int(s.get("wins", 0)),
                    "losses": int(s.get("losses", 0)),
                    "weight": self.weight(rule_id),
                }
            )
        rows.sort(key=lambda r: r["wins"] + r["losses"], reverse=True)
        return {
            "total_rules": len(self.rulebook.rules),
            "immutable": len(self.rulebook.immutable()),
            "learnable": len(self.rulebook.learnable()),
            "tracked": len(self.stats),
            "rules": rows[:top_n],
        }
