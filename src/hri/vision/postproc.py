"""FER post-processing, ported from fer_resnet_demo v2 (chain of `runners.py` webcam + `configs/default.yaml`,
ablation variant V3b = everything except pedagogy):

    softmax
      -> 1. Sensitivity scaling       P'_c = w_c P_c / sum_j w_j P_j            (fer_demo/loss.py)
      -> 2. Personal baseline          sanity-clamped P_base, per-class gains,    (fer_demo/calibration.py)
                                       dynamic neutral suppression
      -> 3. Hysteresis + 2 biomechanical rules: per-class thresholds, t_min,     (fer_demo/temporal.py)
                                       dwell, neutral bridging, transition barriers
    => stable label (hysteresis) + calibrated probabilities -> valence / arousal (circumplex)

Like the demo, EMA is NOT part of the chain any more (vision.smoothing.enabled: false); it can be switched
back on for ablation. Where the demo's README / EmotionPostProcessor disagree with runners.py, runners.py wins
(sensitivity BEFORE baseline, class_gains from config, dwell / bridge 0.25 s).
Detector differs from the demo on purpose: YuNet instead of YOLO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hri.vision.fer import va_from_probs, va_table

# ----------------------------------------------------------------------------- 1. sensitivity scaling
DEFAULT_SENSITIVITY = {"neutral": 0.45, "happiness": 0.95, "surprise": 1.15, "sadness": 1.85,
                       "anger": 1.90, "fear": 3.00, "contempt": 3.40, "disgust": 3.60}


class SensitivityScaler:
    """fer_demo SensitivityScaler: P'_c = w_c P_c / sum_j w_j P_j."""

    def __init__(self, classes: list[str], weights: dict | None = None, enabled: bool = True):
        self.enabled = bool(enabled)
        w = weights or DEFAULT_SENSITIVITY
        self.weights = np.array([float(w.get(c, 1.0)) for c in classes], np.float32)

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, np.float32)
        if not self.enabled:
            return p
        s = p * self.weights
        total = float(s.sum())
        return s / total if total > 0 else p


# ----------------------------------------------------------------------------- 2. personal baseline
DEFAULT_CLASS_GAINS = {"neutral": 1.0, "happiness": 1.6, "surprise": 2.0, "sadness": 3.2, "anger": 3.2,
                       "fear": 3.4, "contempt": 3.4, "disgust": 3.6}


class BaselineCalibrator:
    """fer_demo BaselineCalibrator (Dynamic Gain Control). P_base = mean of the first `calib_seconds` of face
    time, sanity-clamped (non-neutral classes <= max_non_neutral, surplus moved to neutral). Then per frame:
        delta = p - P_base;  m = max(0, max non-neutral delta)
        neutral : max(0, p - neutral_margin - (2.2 * m if dynamic suppression and m > 0.015))
        other   : p + gain_c * delta   if delta > 0   else   p * 0.5
        renormalise."""

    def __init__(self, classes: list[str], calib_seconds: float = 3.0, min_samples: int = 5,
                 max_non_neutral: float = 0.15, neutral_margin: float = 0.20, delta_gain: float = 2.0,
                 class_gains: dict | None = None, dynamic_neutral_suppress: bool = True,
                 adaptive_momentum: float = 0.0, enabled: bool = True):
        self.classes = list(classes)
        self.neutral_idx = self.classes.index("neutral") if "neutral" in self.classes else 0
        self.calib_seconds, self.min_samples = float(calib_seconds), int(min_samples)
        self.max_non_neutral, self.neutral_margin = float(max_non_neutral), float(neutral_margin)
        self.delta_gain = float(delta_gain)
        gains = class_gains or DEFAULT_CLASS_GAINS
        self.gains = np.array([float(gains.get(c, self.delta_gain)) for c in self.classes], np.float32)
        self.dynamic_neutral_suppress = bool(dynamic_neutral_suppress)
        self.adaptive_momentum = float(adaptive_momentum)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.p_base: np.ndarray | None = None
        self.samples: list[np.ndarray] = []
        self.elapsed = 0.0
        self.is_calibrated = False

    @property
    def progress(self) -> float:
        return 1.0 if self.is_calibrated else min(1.0, self.elapsed / max(1e-6, self.calib_seconds))

    def _sanity_clamp(self) -> None:
        if self.p_base is None or self.max_non_neutral <= 0:
            return
        surplus = 0.0
        for i in range(len(self.classes)):
            if i != self.neutral_idx and self.p_base[i] > self.max_non_neutral:
                surplus += float(self.p_base[i] - self.max_non_neutral)
                self.p_base[i] = self.max_non_neutral
        self.p_base[self.neutral_idx] += surplus
        self.p_base = self.p_base / float(self.p_base.sum())

    def update_calibration(self, probs: np.ndarray, dt: float) -> bool:
        """Add one frame. Returns True on the frame that completes calibration."""
        if probs is None or self.is_calibrated:
            return False
        self.samples.append(np.asarray(probs, np.float32))
        self.elapsed += max(0.0, dt)
        if self.elapsed >= self.calib_seconds and len(self.samples) >= self.min_samples:
            base = np.mean(np.stack(self.samples), axis=0)
            self.p_base = base / base.sum()
            self._sanity_clamp()
            self.is_calibrated = True
            return True
        return False

    def adjust(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, np.float32)
        if not self.enabled or not self.is_calibrated or self.p_base is None:
            return probs
        delta = probs - self.p_base
        non_neutral = np.delete(delta, self.neutral_idx)
        max_pos = max(0.0, float(non_neutral.max())) if len(non_neutral) else 0.0
        out = np.where(delta > 0, probs + delta * self.gains, np.maximum(0.0, probs * 0.5)).astype(np.float32)
        penalty = self.neutral_margin
        if self.dynamic_neutral_suppress and max_pos > 0.015:
            penalty += max_pos * 2.2
        out[self.neutral_idx] = max(0.0, float(probs[self.neutral_idx]) - penalty)
        total = float(out.sum())
        out = out / total if total > 1e-6 else probs.copy()
        if self.adaptive_momentum > 0 and int(np.argmax(out)) == self.neutral_idx:
            self.p_base = (1 - self.adaptive_momentum) * self.p_base + self.adaptive_momentum * probs
            self.p_base = self.p_base / self.p_base.sum()
        return out

    def info(self) -> dict:
        return {"enabled": self.enabled, "is_calibrated": self.is_calibrated,
                "p_base": None if self.p_base is None else
                {c: round(float(v), 4) for c, v in zip(self.classes, self.p_base)}}


# ----------------------------------------------------------------------------- 3. hysteresis + biomechanics
POLARITY_GROUPS = {"happiness": "positive", "neutral": "neutral", "surprise": "ambiguous", "sadness": "negative",
                   "anger": "negative", "fear": "negative", "disgust": "negative", "contempt": "negative"}

# Delta_barrier(from -> to), Russell (1980) / Ekman & Friesen (1978) / Cohen et al. (2003) - copied from the demo
DEFAULT_TRANSITION_BARRIERS = {
    "sadness": {"sadness": 0.0, "anger": 0.0, "neutral": 0.0, "contempt": 0.05, "disgust": 0.05, "fear": 0.05,
                "surprise": 0.15, "happiness": 0.35},
    "anger": {"anger": 0.0, "disgust": 0.0, "contempt": 0.0, "sadness": 0.0, "neutral": 0.0, "fear": 0.08,
              "surprise": 0.12, "happiness": 0.35},
    "happiness": {"happiness": 0.0, "surprise": 0.0, "neutral": 0.0, "fear": 0.20, "contempt": 0.25,
                  "anger": 0.30, "disgust": 0.30, "sadness": 0.35},
    "surprise": {"surprise": 0.0, "happiness": 0.05, "fear": 0.05, "neutral": 0.0, "anger": 0.10,
                 "sadness": 0.15, "contempt": 0.15, "disgust": 0.15},
    "disgust": {"disgust": 0.0, "anger": 0.0, "contempt": 0.0, "sadness": 0.05, "neutral": 0.0, "fear": 0.08,
                "surprise": 0.15, "happiness": 0.35},
    "contempt": {"contempt": 0.0, "disgust": 0.0, "anger": 0.0, "sadness": 0.05, "neutral": 0.0, "fear": 0.08,
                 "surprise": 0.15, "happiness": 0.35},
    "fear": {"fear": 0.0, "surprise": 0.05, "sadness": 0.05, "anger": 0.08, "neutral": 0.0, "contempt": 0.10,
             "disgust": 0.10, "happiness": 0.30},
    "neutral": {"neutral": 0.05, "happiness": 0.0, "surprise": 0.0, "sadness": 0.0, "anger": 0.0,
                "disgust": 0.0, "fear": 0.0, "contempt": 0.0},
}
DEFAULT_CLASS_THRESHOLDS = {
    "neutral": {"start": 0.55, "end": 0.38}, "happiness": {"start": 0.38, "end": 0.22},
    "surprise": {"start": 0.32, "end": 0.18}, "sadness": {"start": 0.24, "end": 0.12},
    "anger": {"start": 0.24, "end": 0.12}, "fear": {"start": 0.18, "end": 0.08},
    "contempt": {"start": 0.18, "end": 0.08}, "disgust": {"start": 0.18, "end": 0.08},
}


class HysteresisSegmenter:
    """fer_demo HysteresisSegmenter (temporal.py), same decision logic; session-segment export removed.

    - Challenger = class (!= active) exceeding its effective start threshold by the largest ratio
      (not argmax, which the neutral bias would win).
    - Effective start threshold = per-class start + transition barrier(active -> candidate).   [rule 2]
    - Dwell: a non-neutral state cannot be left before min_dwell_s unless candidate >= 0.85.     [rule 1]
    - Neutral bridge: negative <-> positive must stay in neutral >= neutral_bridge_s (unless >= 0.85).
    - Switch after the candidate holds for t_min; fall back when the active class drops below its end
      threshold for longer than fade_grace_s."""

    def __init__(self, classes: list[str], theta_start: float = 0.55, theta_end: float = 0.35,
                 t_min: float = 0.25, min_dwell_s: float = 0.25, neutral_bridge_s: float = 0.25,
                 fade_grace_s: float = 0.20, transition_dynamics: bool = True, barrier_opposing: float = 0.35,
                 barrier_distant: float = 0.15, transition_barriers: dict | None = None,
                 class_thresholds: dict | None = None, default_neutral: str = "neutral", enabled: bool = True):
        self.classes = list(classes)
        self.theta_start, self.theta_end = float(theta_start), float(theta_end)
        self.t_min, self.min_dwell_s = float(t_min), float(min_dwell_s)
        self.neutral_bridge_s, self.fade_grace_s = float(neutral_bridge_s), float(fade_grace_s)
        self.transition_dynamics = bool(transition_dynamics)
        self.barrier_opposing, self.barrier_distant = float(barrier_opposing), float(barrier_distant)
        self.transition_barriers = dict(transition_barriers) if transition_barriers else DEFAULT_TRANSITION_BARRIERS
        self.class_thresholds = class_thresholds or DEFAULT_CLASS_THRESHOLDS
        self.default_neutral = default_neutral if default_neutral in self.classes else self.classes[0]
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.active_state: str | None = None
        self.state_start_t = 0.0
        self.candidate_state: str | None = None
        self.candidate_start_t = 0.0
        self.last_non_neutral_state: str | None = None
        self.neutral_start_t: float | None = None
        self.fade_start_t: float | None = None

    def thresholds(self, emotion: str) -> tuple[float, float]:
        th = (self.class_thresholds or {}).get(emotion)
        if th:
            return float(th.get("start", self.theta_start)), float(th.get("end", self.theta_end))
        return self.theta_start, self.theta_end

    def effective_start(self, cand: str) -> float:
        base, _ = self.thresholds(cand)
        if not self.transition_dynamics or not self.active_state:
            return base
        if self.active_state in self.transition_barriers:
            barrier = float(self.transition_barriers[self.active_state].get(cand, 0.0))
        else:
            fp = POLARITY_GROUPS.get(self.active_state, "neutral")
            tp = POLARITY_GROUPS.get(cand, "neutral")
            if {fp, tp} == {"negative", "positive"}:
                barrier = self.barrier_opposing
            elif fp != tp and "neutral" not in (fp, tp):
                barrier = self.barrier_distant
            else:
                barrier = 0.0
        return min(0.95, base + barrier)

    def is_buffer(self, t: float) -> bool:
        """In the neutral rest right after an expression (EmotionPostProcessor.is_buffer)."""
        return (self.active_state == self.default_neutral and self.last_non_neutral_state is not None
                and self.neutral_start_t is not None and t - self.neutral_start_t < self.neutral_bridge_s)

    def _switch(self, new_state: str, t_start: float) -> None:
        if self.active_state != self.default_neutral:
            self.last_non_neutral_state = self.active_state
        self.active_state, self.state_start_t = new_state, t_start
        self.fade_start_t = None
        if new_state == self.default_neutral:
            self.neutral_start_t = t_start

    def update(self, t: float, probs: np.ndarray | None) -> str:
        if probs is None or not self.enabled:
            if not self.enabled and probs is not None:
                return self.classes[int(np.argmax(probs))]
            return self.active_state or self.default_neutral
        probs = np.asarray(probs, np.float32)

        if self.active_state is None:
            top = self.classes[int(np.argmax(probs))]
            self.active_state, self.state_start_t = top, t
            if top == self.default_neutral:
                self.neutral_start_t = t
            else:
                self.last_non_neutral_state = top
            return top

        active_prob = float(probs[self.classes.index(self.active_state)])

        # 1. challenger with the largest ratio over its effective threshold
        best, best_p, best_ratio = None, 0.0, 0.0
        for i, c in enumerate(self.classes):
            if c == self.active_state:
                continue
            th = self.effective_start(c)
            if probs[i] >= th and probs[i] / max(1e-4, th) > best_ratio:
                best, best_p, best_ratio = c, float(probs[i]), float(probs[i]) / max(1e-4, th)
        if best is None:
            k = int(np.argmax(probs))
            best, best_p = self.classes[k], float(probs[k])
        cand, cand_p, cand_th = best, best_p, self.effective_start(best)

        # 2. rule 1: dwell + neutral bridging
        can_switch = True
        if self.transition_dynamics and cand != self.active_state:
            if self.active_state != self.default_neutral and t - self.state_start_t < self.min_dwell_s \
                    and cand_p < 0.85:
                can_switch = False
            if self.active_state == self.default_neutral and self.last_non_neutral_state is not None:
                pp = POLARITY_GROUPS.get(self.last_non_neutral_state, "neutral")
                cp = POLARITY_GROUPS.get(cand, "neutral")
                if {pp, cp} == {"negative", "positive"}:
                    n_start = self.neutral_start_t if self.neutral_start_t is not None else self.state_start_t
                    if t - n_start < self.neutral_bridge_s and cand_p < 0.85:
                        can_switch = False

        if cand != self.active_state and cand_p >= cand_th and can_switch:
            if self.candidate_state == cand:
                if t - self.candidate_start_t >= self.t_min:
                    self._switch(cand, self.candidate_start_t)
                    self.candidate_state = None
            else:
                self.candidate_state, self.candidate_start_t = cand, t
            return self.active_state

        # 3. no challenger: decay of the active state (with grace period)
        self.candidate_state = None
        _, end_th = self.thresholds(self.active_state)
        if active_prob < end_th:
            if self.active_state != self.default_neutral:
                if self.fade_start_t is None:
                    self.fade_start_t = t
                if t - self.fade_start_t < self.fade_grace_s:
                    return self.active_state
            fallback = cand if (cand_p >= cand_th and can_switch and cand != self.active_state) \
                else self.default_neutral
            if fallback != self.active_state:
                self._switch(fallback, t)
        else:
            self.fade_start_t = None
        return self.active_state


# ----------------------------------------------------------------------------- stream (per user)
@dataclass
class AffectFrame:
    t: float
    raw: np.ndarray            # model softmax
    probs: np.ndarray          # after sensitivity (+ baseline once calibrated)
    label: str                 # stable label (hysteresis + biomechanical rules)
    top: str                   # argmax of the calibrated probabilities (before hysteresis)
    confidence: float          # calibrated probability of the stable label
    valence: float             # from the calibrated probabilities
    arousal: float
    quality: float
    calibrated: bool
    buffer: bool = False       # neutral rest between two expressions

    def to_msg(self, labels: list[str]) -> dict:
        """Dict format consumed by AffectAggregator.add()."""
        return {"t": self.t, "probs": self.probs.tolist(), "labels": labels, "label": self.label,
                "valence": self.valence, "arousal": self.arousal, "confidence": self.confidence,
                "quality": self.quality}


class EMASmoother:
    """Optional (vision.smoothing.enabled) - the demo keeps EMA only as a comparator, not in the chain."""

    def __init__(self, alpha: float = 0.45):
        self.alpha = float(alpha)
        self.ema: np.ndarray | None = None

    def reset(self) -> None:
        self.ema = None

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        self.ema = probs.copy() if self.ema is None else self.alpha * probs + (1 - self.alpha) * self.ema
        return self.ema


class FERStream:
    """Stateful per-user stream: feed vision messages (raw softmax), get AffectFrames.
    While the baseline is being measured, frames are still returned (uncalibrated)."""

    def __init__(self, classes: list[str], vision_cfg: dict):
        self.classes = list(classes)
        s = vision_cfg.get("sensitivity", {}) or {}
        b = vision_cfg.get("baseline", {}) or {}
        h = vision_cfg.get("hysteresis", {}) or {}
        sm = vision_cfg.get("smoothing", {}) or {}
        self.scaler = SensitivityScaler(self.classes, s.get("weights"), s.get("enabled", True))
        self.calibrator = BaselineCalibrator(
            self.classes, b.get("calib_seconds", 3.0), b.get("min_samples", 5), b.get("max_non_neutral", 0.15),
            b.get("neutral_margin", 0.20), b.get("delta_gain", 2.0), b.get("class_gains"),
            b.get("dynamic_neutral_suppress", True), b.get("adaptive_momentum", 0.0), b.get("enabled", True))
        self.segmenter = HysteresisSegmenter(
            self.classes, h.get("theta_start", 0.55), h.get("theta_end", 0.35), h.get("t_min", 0.25),
            h.get("min_dwell_s", 0.25), h.get("neutral_bridge_s", 0.25), h.get("fade_grace_s", 0.20),
            h.get("transition_dynamics", True), h.get("barrier_opposing", 0.35), h.get("barrier_distant", 0.15),
            None, h.get("class_thresholds"), "neutral", h.get("enabled", True))
        self.smoother = EMASmoother(sm.get("alpha", 0.45)) if sm.get("enabled", False) else None
        self.reset_after = int(sm.get("reset_after", 10))
        # the demo adds one frame period for every calibration frame, including the first one
        self.nominal_dt = 1.0 / float(vision_cfg.get("target_fps", 10) or 10)
        self._va = va_table(self.classes)
        self._last_t: float | None = None
        self._missing = 0
        self.face_seen = False
        self.calibrating = False

    def start_calibration(self) -> None:
        self.calibrator.reset()
        self.segmenter.reset()
        self.calibrating = self.calibrator.enabled

    def cancel_calibration(self) -> None:
        """Person left / took too long: go on uncalibrated instead of waiting forever."""
        self.calibrating = False

    def update(self, msg: dict) -> AffectFrame | None:
        if msg.get("type") != "fer":
            self._last_t = None
            self._missing += 1
            if self._missing > self.reset_after:     # person left: next one starts a fresh state
                self.segmenter.reset()
                if self.smoother:
                    self.smoother.reset()
            return None
        self._missing = 0
        self.face_seen = True
        t = float(msg["t"])
        # face time: one nominal period for the first frame after a gap, gaps themselves don't count
        dt = self.nominal_dt if self._last_t is None else min(0.5, t - self._last_t)
        self._last_t = t
        raw = np.asarray(msg["probs"], np.float32)
        p = self.scaler(raw)                                                # 1. sensitivity
        if self.smoother:
            p = self.smoother(p)
        if self.calibrating and self.calibrator.update_calibration(p, dt):  # 2. baseline (on scaled probs)
            self.calibrating = False
        probs = self.calibrator.adjust(p)
        label = self.segmenter.update(t, probs)                             # 3. hysteresis + biomechanics
        v, a = va_from_probs(probs, self._va)
        return AffectFrame(t, raw, probs, label, self.classes[int(np.argmax(probs))],
                           float(probs[self.classes.index(label)]), v, a, float(msg.get("quality", 1.0)),
                           self.calibrator.is_calibrated, self.segmenter.is_buffer(t))
