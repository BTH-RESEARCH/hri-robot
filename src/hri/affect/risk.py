"""Intervention logic: local rules that run before / instead of the LLM (no network dependency)."""
from __future__ import annotations

from hri.affect.child_state import CONFUSED, DISLIKE, ChildQueue


def adult_crisis(text: str, safety_cfg: dict) -> bool:
    """The speaking parent mentions self-harm (keyword rule)."""
    t = text.lower()
    return bool(safety_cfg.get("enabled", True)) and any(k in t for k in safety_cfg.get("keywords", []))


def child_distress(queue: ChildQueue, now: float, cfg: dict) -> bool:
    """The child's face looked upset (sad / afraid / angry / disgusted) for several seconds."""
    return bool(cfg.get("enabled", True)) and queue.sustained({CONFUSED, DISLIKE}, now, float(cfg.get("seconds", 4.0)),
                                                              float(cfg.get("ratio", 0.8)))
