"""Micro-loop: every second the robot's face reacts to the child's group in the last window (no LLM, no speech)."""
from __future__ import annotations

from dataclasses import dataclass

from hri.affect.child_state import CONFUSED, DISLIKE, ENJOY, FOCUS

REACTION = {FOCUS: "listening", ENJOY: "happy", CONFUSED: "empathy", DISLIKE: "calm"}
BY_LABEL = {"surprise": "curious", "fear": "concerned"}   # finer reaction inside a group
AWAY = "curious"                                          # face missing: look for the child


@dataclass
class MicroDecision:
    group: str
    expression: str
    intensity: float
    reason: str        # "face" | "away"

    def to_dict(self) -> dict:
        return {"group": self.group, "expression": self.expression, "intensity": round(self.intensity, 2),
                "reason": self.reason}


def decide(summary: dict, cfg: dict) -> MicroDecision | None:
    lo, hi = float(cfg.get("min_intensity", 0.3)), float(cfg.get("max_intensity", 0.8))
    if not summary.get("visible"):
        return MicroDecision(DISLIKE, AWAY, lo, "away") if summary.get("reason") == "away" else None
    if summary["dominance"] < float(cfg.get("min_dominance", 0.5)):
        return None                       # no clear state: keep the current face
    g = summary["group"]
    if g == FOCUS:
        return MicroDecision(g, REACTION[g], 0.5, "face")
    strength = summary["mean_probs"][g] * (1.0 if summary["calibrated"] else 0.5)
    return MicroDecision(g, BY_LABEL.get(summary["label"], REACTION[g]), lo + (hi - lo) * min(1.0, strength), "face")


def pace(summary: dict, cfg: dict) -> float:
    """Multiplier of the TTS speed for the next sentence."""
    p = cfg.get("pace", {}) or {}
    if summary.get("visible") and not summary.get("calibrated"):
        return 1.0
    return float(p.get(summary.get("group", FOCUS), 1.0))
