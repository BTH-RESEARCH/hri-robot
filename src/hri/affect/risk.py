"""Intervention logic: local, rule-based, runs BEFORE the LLM (no network dependency)."""
from __future__ import annotations


def assess_risk(text: str, affect: dict, safety_cfg: dict) -> str:
    if not safety_cfg.get("enabled", True):
        return "low"
    t = text.lower()
    if any(k in t for k in safety_cfg.get("keywords", [])):
        return "high"
    if affect.get("face_visible") and affect.get("valence", 0) < safety_cfg.get("negative_valence_threshold", -0.6) \
            and (affect.get("stability") or 0) > 0.6:   # stability is None when ES cannot be computed
        return "elevated"
    return "low"
