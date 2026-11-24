"""Micro-Loop (Fluid-Xpress, Sec. 3.1): non-verbal feedback on a regular sliding window.

Every `update_sec` the robot looks at the last `window_sec` of calibrated FER frames, computes the
quality-weighted mean valence / arousal and picks a strategy:

    De-escalation : arousal >= high_arousal and valence not positive -> calm face
    Mirroring     : valence >= positive_valence                     -> mirror the user's positive face
    Support       : valence <= negative_valence                     -> empathic / concerned face
    Attentive     : otherwise                                       -> listening face

Thresholds live in configs (micro_loop.*). Pure function `decide()` so it can be unit-tested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from hri.affect.aggregator import weighted_label_vote

MIRRORING, SUPPORT, DEESCALATION, ATTENTIVE = "mirroring", "support", "de-escalation", "attentive"

# user's dominant FERPlus class -> robot expression, per strategy
MIRROR_EXPR = {"happiness": "happy", "surprise": "surprised"}
SUPPORT_EXPR = {"sadness": "empathy", "fear": "concerned", "disgust": "concerned",
                "contempt": "concerned", "anger": "concerned"}
DEFAULT_EXPR = {MIRRORING: "happy", SUPPORT: "empathy", DEESCALATION: "calm", ATTENTIVE: "listening"}


@dataclass
class MicroDecision:
    strategy: str
    expression: str
    intensity: float
    valence: float
    arousal: float
    dominant: str
    n_frames: int

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("intensity", "valence", "arousal"):
            d[k] = round(d[k], 3)
        return d


def _scale(magnitude: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(np.clip(magnitude, 0.0, 1.0))


def decide(frames: list, labels: list[str], cfg: dict) -> MicroDecision | None:
    """frames: objects with .probs .valence .arousal .quality (AffectAggregator.Frame)."""
    if len(frames) < int(cfg.get("min_frames", 3)):
        return None
    w = np.array([max(1e-3, f.quality) for f in frames], np.float64)
    w /= w.sum()
    v = float(w @ np.array([f.valence for f in frames]))
    a = float(w @ np.array([f.arousal for f in frames]))
    dist = (w[:, None] * np.stack([f.probs for f in frames])).sum(0)
    # the user's stable (hysteresis) label picks WHICH face to mirror / support; V/A pick the strategy
    dominant = weighted_label_vote(frames, w) or labels[int(np.argmax(dist))]

    pos, neg = float(cfg.get("positive_valence", 0.2)), float(cfg.get("negative_valence", -0.2))
    high_a = float(cfg.get("high_arousal", 0.5))
    lo, hi = float(cfg.get("min_intensity", 0.3)), float(cfg.get("max_intensity", 0.8))

    if a >= high_a and v < pos:
        strategy, expr, mag = DEESCALATION, DEFAULT_EXPR[DEESCALATION], (a - high_a) / max(1e-6, 1 - high_a)
    elif v >= pos:
        strategy, mag = MIRRORING, (v - pos) / max(1e-6, 1 - pos)
        expr = MIRROR_EXPR.get(dominant, DEFAULT_EXPR[MIRRORING])
    elif v <= neg:
        strategy, mag = SUPPORT, (neg - v) / max(1e-6, 1 + neg)
        expr = SUPPORT_EXPR.get(dominant, DEFAULT_EXPR[SUPPORT])
    else:
        strategy, expr, mag = ATTENTIVE, DEFAULT_EXPR[ATTENTIVE], 0.5
    return MicroDecision(strategy, expr, _scale(mag, lo, hi), v, a, dominant, len(frames))
