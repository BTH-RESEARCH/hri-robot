"""Temporal post-processing of FER probabilities, ported from fer_resnet_demo (v2 pipeline):

    raw probs -> EMA smoothing (fer_demo/pipeline.TemporalSmoother, method=ema)
              -> personal baseline calibration (fer_demo/calibration.BaselineCalibrator)
              -> valence / arousal (circumplex lookup)

Sensitivity scaling, hysteresis segmentation and pedagogy are intentionally NOT used in the HRI system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hri.vision.fer import va_from_probs, va_table


class EMASmoother:
    """fer_demo TemporalSmoother (method=ema): reset after `reset_after` frames without a face."""

    def __init__(self, alpha: float = 0.3, reset_after: int = 10):
        self.alpha, self.reset_after = float(alpha), int(reset_after)
        self.reset()

    def reset(self) -> None:
        self.ema: np.ndarray | None = None
        self.missing = 0

    def update(self, probs: np.ndarray | None) -> np.ndarray | None:
        if probs is None:
            self.missing += 1
            if self.missing > self.reset_after:
                self.reset()
            return None
        self.missing = 0
        probs = np.asarray(probs, np.float32)
        self.ema = probs.copy() if self.ema is None else self.alpha * probs + (1 - self.alpha) * self.ema
        return self.ema


class BaselineCalibrator:
    """fer_demo BaselineCalibrator. Collects P_base over `calib_seconds` of face time, then:
        neutral      -> max(0, p - neutral_margin)
        other class  -> p + gain * (p - P_base)   if above the user's baseline
                        p * 0.5                    otherwise
        renormalise.
    (This is the behaviour of the demo's code; the README describes a plain S_t - P_base.)"""

    def __init__(self, classes: list[str], calib_seconds: float = 3.0, min_samples: int = 5,
                 neutral_margin: float = 0.15, gain: float = 1.5, adaptive_momentum: float = 0.0,
                 enabled: bool = True):
        self.classes = list(classes)
        self.neutral_idx = self.classes.index("neutral") if "neutral" in self.classes else 0
        self.calib_seconds, self.min_samples = float(calib_seconds), int(min_samples)
        self.neutral_margin, self.gain = float(neutral_margin), float(gain)
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

    def update_calibration(self, probs: np.ndarray, dt: float) -> bool:
        """Add one frame. Returns True on the frame that completes calibration."""
        if probs is None or self.is_calibrated:
            return False
        self.samples.append(np.asarray(probs, np.float32))
        self.elapsed += max(0.0, dt)
        if self.elapsed >= self.calib_seconds and len(self.samples) >= self.min_samples:
            base = np.mean(np.stack(self.samples), axis=0)
            self.p_base = base / base.sum()
            self.is_calibrated = True
            return True
        return False

    def adjust(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, np.float32)
        if not self.enabled or not self.is_calibrated or self.p_base is None:
            return probs
        delta = probs - self.p_base
        out = np.where(delta > 0, probs + delta * self.gain, probs * 0.5).astype(np.float32)
        out[self.neutral_idx] = max(0.0, float(probs[self.neutral_idx]) - self.neutral_margin)
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


@dataclass
class AffectFrame:
    t: float
    raw: np.ndarray            # model softmax
    probs: np.ndarray          # after EMA (+ baseline once calibrated)
    label: str
    confidence: float
    valence: float
    arousal: float
    quality: float
    calibrated: bool

    def to_msg(self, labels: list[str]) -> dict:
        """Dict format consumed by AffectAggregator.add()."""
        return {"t": self.t, "probs": self.probs.tolist(), "labels": labels, "valence": self.valence,
                "arousal": self.arousal, "confidence": self.confidence, "quality": self.quality}


class FERStream:
    """Stateful per-user stream: feed vision messages, get calibrated AffectFrames.
    While the baseline is being measured, frames are still returned (smoothed, uncalibrated)."""

    def __init__(self, classes: list[str], vision_cfg: dict):
        s = vision_cfg.get("smoothing", {}) or {}
        b = vision_cfg.get("baseline", {}) or {}
        self.classes = list(classes)
        self.smoother = EMASmoother(s.get("alpha", 0.3), s.get("reset_after", 10))
        self.calibrator = BaselineCalibrator(
            self.classes, b.get("calib_seconds", 3.0), b.get("min_samples", 5), b.get("neutral_margin", 0.15),
            b.get("gain", 1.5), b.get("adaptive_momentum", 0.0), b.get("enabled", True))
        self._va = va_table(self.classes)
        self._last_t: float | None = None
        self.calibrating = False

    def start_calibration(self) -> None:
        self.calibrator.reset()
        self.calibrating = self.calibrator.enabled

    def update(self, msg: dict) -> AffectFrame | None:
        if msg.get("type") != "fer":
            self.smoother.update(None)
            self._last_t = None
            return None
        t = float(msg["t"])
        dt = 0.0 if self._last_t is None else min(0.5, t - self._last_t)   # gaps don't count as face time
        self._last_t = t
        raw = np.asarray(msg["probs"], np.float32)
        smooth = self.smoother.update(raw)
        if self.calibrating and self.calibrator.update_calibration(smooth, dt):
            self.calibrating = False
        probs = self.calibrator.adjust(smooth)
        k = int(np.argmax(probs))
        v, a = va_from_probs(probs, self._va)
        return AffectFrame(t, raw, probs, self.classes[k], float(probs[k]), v, a,
                           float(msg.get("quality", 1.0)), self.calibrator.is_calibrated)
