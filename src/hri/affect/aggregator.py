"""Deep Affective Engine: turns a stream of per-frame FER results into an utterance-aligned
affect summary for the LLM.

Key ideas (research contributions you can ablate):
  1. Utterance alignment  - only frames in [speech_start - pre, speech_end] count.
  2. Speaking down-weight - mouth motion during speech confuses FER.
  3. Quality weighting    - detector score, face size, sharpness, model confidence.
  4. Per-session baseline - valence relative to the current person's usual face.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    t: float
    probs: np.ndarray
    valence: float
    arousal: float
    confidence: float
    quality: float
    label: str | None = None    # stable label from the hysteresis segmenter


def weighted_label_vote(frames: list, weights: np.ndarray | None = None) -> str | None:
    """Most common stable (hysteresis) label in a window, weighted; None if frames carry no labels."""
    votes: dict[str, float] = {}
    for i, f in enumerate(frames):
        if getattr(f, "label", None):
            votes[f.label] = votes.get(f.label, 0.0) + (1.0 if weights is None else float(weights[i]))
    return max(votes, key=votes.get) if votes else None


class AffectAggregator:
    def __init__(self, affect_cfg: dict, labels: list[str]):
        self.cfg = affect_cfg
        self.labels = labels
        self.frames: deque[Frame] = deque()
        self.speaking: deque[tuple[float, float]] = deque(maxlen=200)
        self.baselines: dict[str, tuple[float, int]] = {}   # session -> (valence EMA, n)
        self.last_face_t = 0.0
        self._lock = threading.Lock()

    # ---------- input ----------
    def add(self, msg: dict) -> None:
        f = Frame(msg["t"], np.asarray(msg["probs"], float), msg["valence"], msg["arousal"],
                  msg["confidence"], msg["quality"], msg.get("label"))
        with self._lock:
            self.frames.append(f)
            self.last_face_t = f.t
            horizon = f.t - float(self.cfg.get("buffer_sec", 60))
            while self.frames and self.frames[0].t < horizon:
                self.frames.popleft()

    def reset(self) -> None:
        """New person: forget frames, speaking intervals and the valence baseline."""
        with self._lock:
            self.frames.clear()
            self.speaking.clear()
            self.baselines.clear()
            self.last_face_t = 0.0

    def mark_speaking(self, t0: float, t1: float) -> None:
        with self._lock:
            self.speaking.append((t0, t1))

    def frames_between(self, t0: float, t1: float) -> list[Frame]:
        with self._lock:
            return [f for f in self.frames if t0 <= f.t <= t1]

    def latest(self, max_age: float = 1.0) -> Frame | None:
        with self._lock:
            if self.frames and self.frames[-1].t >= time.time() - max_age:
                return self.frames[-1]
        return None

    # ---------- summary ----------
    def _is_speaking(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self.speaking)

    def summarize(self, t0: float, t1: float, person: str | None = None) -> dict:
        pre = float(self.cfg.get("pre_utterance_sec", 1.0))
        with self._lock:
            sel = [f for f in self.frames if t0 - pre <= f.t <= t1]
        if len(sel) < int(self.cfg.get("min_frames", 3)):
            return {"face_visible": False}

        sw = float(self.cfg.get("speaking_weight", 0.3))
        w = np.array([f.quality * f.confidence * (sw if self._is_speaking(f.t) else 1.0) for f in sel])
        w = w / w.sum() if w.sum() > 0 else np.full(len(sel), 1 / len(sel))
        probs = np.stack([f.probs for f in sel])
        dist = (w[:, None] * probs).sum(0)
        val = float((w * [f.valence for f in sel]).sum())
        aro = float((w * [f.arousal for f in sel]).sum())

        ts = np.array([f.t for f in sel])
        vs = np.array([f.valence for f in sel])
        slope = float(np.polyfit(ts - ts[0], vs, 1)[0]) if np.ptp(ts) > 0.5 else 0.0
        thr = float(self.cfg.get("trend_threshold", 0.05))
        trend = "rising" if slope > thr else "falling" if slope < -thr else "stable"

        es = emotional_stability(ts, vs, np.array([f.arousal for f in sel]), self.cfg.get("stability", {}) or {})

        top = np.argsort(dist)[::-1][:3]
        out = {
            "face_visible": True,
            # stable hysteresis label (weighted vote); falls back to the probability argmax
            "dominant": weighted_label_vote(sel, w) or self.labels[int(top[0])],
            "dist": {self.labels[i]: round(float(dist[i]), 2) for i in top},
            "valence": round(val, 2),
            "arousal": round(aro, 2),
            "trend": trend,
            "stability": None if es is None else round(es["es"], 2),
            "confidence": round(float(np.mean([f.confidence for f in sel])), 2),
            # internal (logged, not sent to the LLM)
            "n_frames": len(sel),
            "mssd": None if es is None else round(es["mssd"], 4),
            "pac": None if es is None else round(es["pac"], 3),
        }
        if person:
            out["vs_baseline"] = self._baseline_delta(person, val)
        return out

    def _baseline_delta(self, person: str, val: float) -> str:
        alpha = float(self.cfg.get("baseline_alpha", 0.05))
        base, n = self.baselines.get(person, (val, 0))
        delta = val - base if n >= 3 else 0.0
        self.baselines[person] = (base + alpha * (val - base) if n else val, n + 1)
        if n < 3:
            return "unknown"
        return "more_positive" if delta > 0.2 else "more_negative" if delta < -0.2 else "typical"


def _segments(ts: np.ndarray, xs: np.ndarray, hz: float, max_gap: float) -> list[np.ndarray]:
    """Split at face-loss gaps, then (optionally) resample each piece to a fixed rate so MSSD/PAC
    do not depend on the device's FPS (laptop ~15 FPS vs Pi5 ~8 FPS)."""
    cuts = np.where(np.diff(ts) > max_gap)[0] + 1
    out = []
    for t_seg, x_seg in zip(np.split(ts, cuts), np.split(xs, cuts)):
        if len(t_seg) < 2:
            continue
        if hz > 0:
            grid = np.arange(t_seg[0], t_seg[-1] + 1e-9, 1.0 / hz)
            if len(grid) < 2:
                continue
            x_seg = np.interp(grid, t_seg, x_seg)
        out.append(x_seg)
    return out


def mssd_pac(segments: list[np.ndarray], acute: float) -> tuple[float, float] | None:
    """MSSD = mean squared successive difference; PAC = share of successive |diff| >= acute (Jahng et al.)."""
    diffs = np.concatenate([np.diff(s) for s in segments]) if segments else np.zeros(0)
    if len(diffs) == 0:
        return None
    return float(np.mean(diffs ** 2)), float(np.mean(np.abs(diffs) >= acute))


def emotional_stability(ts: np.ndarray, valence: np.ndarray, arousal: np.ndarray, cfg: dict) -> dict | None:
    """Fluid-Xpress Eq. 2: ES = max(0, 1 - 1/2 [min(MSSD_c/M_max, 1) + min(PAC_c/P_max, 1)]),
    MSSD_c / PAC_c averaged over valence and arousal. M_max=0.1, P_max=0.3 are the paper's values
    (calibrated for EmoNet) - recalibrate for this FER model. The acute-change threshold is not given
    in the paper (config: affect.stability.acute_threshold)."""
    hz, gap = float(cfg.get("resample_hz", 5)), float(cfg.get("max_gap_sec", 0.5))
    mv = mssd_pac(_segments(ts, valence, hz, gap), float(cfg.get("acute_threshold", 0.15)))
    ma = mssd_pac(_segments(ts, arousal, hz, gap), float(cfg.get("acute_threshold", 0.15)))
    if mv is None or ma is None:
        return None
    mssd_c, pac_c = (mv[0] + ma[0]) / 2, (mv[1] + ma[1]) / 2
    m_max, p_max = float(cfg.get("m_max", 0.1)), float(cfg.get("p_max", 0.3))
    es = max(0.0, 1 - 0.5 * (min(mssd_c / m_max, 1.0) + min(pac_c / p_max, 1.0)))
    return {"es": es, "mssd": mssd_c, "pac": pac_c}


def congruence(face_valence: float | None, text_valence: float | None, threshold: float) -> str:
    if face_valence is None or text_valence is None:
        return "unknown"
    diff = abs(face_valence - text_valence)
    opposite = np.sign(face_valence) != np.sign(text_valence) and min(abs(face_valence), abs(text_valence)) > 0.15
    return "incongruent" if diff > threshold and opposite else "congruent"
