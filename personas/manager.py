"""Risk & Portfolio manager agents for the Partners' Room.

Deterministic counterparts of ai-hedge-fund's risk/portfolio managers:

- RiskManagerAgent: re-states the engine's IMMUTABLE rule vetoes and hard
  position limits as a persona-style review. It never originates risk logic
  of its own — the RuleBrain verdict and the engine's caps are the law; the
  risk desk merely reads the charges aloud and stamps the page.
- PortfolioManagerAgent: confidence-weighted consensus over the persona
  votes, each vote weighted by that persona's historical accuracy
  (personas/persona_stats.json; default weight 1.0). Emits the final
  BUY / SELL / HOLD, consensus strength, and a dissent tally.
"""

from __future__ import annotations

from knowledge.brain import RuleVerdict
from personas.engine import BEARISH, BULLISH, NEUTRAL, PersonaStats
from utils.logger import get_logger

log = get_logger("personas.manager")

RISK_MANAGER_NAME = "The Risk Desk"
PORTFOLIO_MANAGER_NAME = "The Managing Partner"

# Consensus must clear this before the PM calls a direction.
CONSENSUS_THRESHOLD = 0.15
# Persona consensus may move the setup score by at most this many points.
MAX_SCORE_MODIFIER = 15


class RiskManagerAgent:
    """Applies IMMUTABLE vetoes + position limits; reports as a persona."""

    persona = RISK_MANAGER_NAME

    def review(self, setup: dict, context: dict,
               verdict: RuleVerdict | None = None) -> dict:
        vetoes: list[str] = list(verdict.veto_citations) if verdict else []
        notes: list[str] = []

        # Hard position limit (mirrors AutonomousEngine.open_position's gate).
        open_n = context.get("open_positions")
        max_c = context.get("max_concurrent")
        if (max_c and open_n is not None and open_n >= max_c
                and not context.get("existing_position_unrealized")):
            vetoes.append(
                f"Position limit: {open_n}/{max_c} book slots filled — no new names")

        dd = context.get("drawdown_pct")
        if dd is not None:
            notes.append(f"Book drawdown {dd:.2f}% of peak")
        gross = context.get("gross_exposure_pct")
        if gross is not None:
            notes.append(f"Gross exposure {gross:.1f}% of equity")
        risk = context.get("planned_risk_pct")
        if risk is not None:
            notes.append(f"Planned risk {risk:.2f}% of equity on this name")

        vetoed = bool(vetoes)
        return {
            "persona": self.persona,
            "signal": BEARISH if vetoed else NEUTRAL,
            "confidence": 100 if vetoed else 60,
            "vetoed": vetoed,
            "veto_citations": vetoes,
            "reasoning": vetoes + notes,
            "quote": ("The risk desk holds absolute veto — and today it uses it."
                      if vetoed else
                      "No objection from the risk desk; stay inside the house limits."),
            "dominant_factor": "veto" if vetoed else "clearance",
        }


class PortfolioManagerAgent:
    """Synthesizes the partners' votes into one weighted decision."""

    persona = PORTFOLIO_MANAGER_NAME

    def __init__(self, stats: PersonaStats | None = None):
        self.stats = stats or PersonaStats()

    def synthesize(self, setup: dict, votes: list[dict],
                   risk_review: dict | None = None) -> dict:
        """Weighted consensus: sum of confidence-weighted directional votes,
        weight = persona's historical accuracy (default 1.0)."""
        bulls, bears, abstains = [], [], []
        num = den = 0.0
        for v in votes:
            persona = v.get("persona", "?")
            signal = v.get("signal", NEUTRAL)
            conf = float(v.get("confidence", 0)) / 100.0
            if signal == BULLISH:
                bulls.append(persona)
            elif signal == BEARISH:
                bears.append(persona)
            else:
                abstains.append(persona)
                continue
            w = self.stats.weight(persona)
            sign = 1.0 if signal == BULLISH else -1.0
            num += sign * conf * w
            den += conf * w

        consensus = (num / den) if den > 0 else 0.0
        strength = int(round(abs(consensus) * 100))
        vetoed = bool(risk_review and risk_review.get("vetoed"))

        if vetoed:
            action = "HOLD"
        elif consensus >= CONSENSUS_THRESHOLD:
            action = "BUY"
        elif consensus <= -CONSENSUS_THRESHOLD:
            action = "SELL"
        else:
            action = "HOLD"

        modifier = 0 if vetoed else int(round(
            max(-1.0, min(1.0, consensus)) * MAX_SCORE_MODIFIER))

        dissent = (f"{len(bulls)} bull{'s' if len(bulls) != 1 else ''}, "
                   f"{len(bears)} bear{'s' if len(bears) != 1 else ''}, "
                   f"{len(abstains)} abstain")

        if vetoed:
            summary = ("The risk desk has stamped its veto — the partners may "
                       "argue, but the answer is no.")
        elif action == "HOLD":
            summary = (f"The room is split ({dissent}); without a working "
                       f"majority the house does nothing.")
        else:
            lead = bulls if action == "BUY" else bears
            summary = (f"The partners lean {action} on {setup.get('symbol', '?')} "
                       f"at {strength}% consensus ({dissent}); "
                       f"loudest voice: {lead[0] if lead else 'none'}.")

        return {
            "persona": self.persona,
            "action": action,
            "consensus": round(consensus, 4),
            "strength": strength,
            "dissent": dissent,
            "bulls": bulls,
            "bears": bears,
            "abstains": abstains,
            "score_modifier": modifier,
            "vetoed": vetoed,
            "veto_citations": list(risk_review.get("veto_citations", [])) if risk_review else [],
            "summary": summary,
        }
