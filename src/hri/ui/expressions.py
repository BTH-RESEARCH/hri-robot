"""Parametric face: every expression is a point in parameter space. Intensity lerps from neutral,
transitions are tweened by the renderer, so any [EMO:x|0.3] looks natural."""
from __future__ import annotations

import threading
from dataclasses import dataclass, fields, replace


@dataclass
class FaceParams:
    eye_open: float = 0.9     # 0 closed .. 1 wide
    eye_happy: float = 0.0    # lower lid raised (smiling eyes)
    eye_scale: float = 1.0
    brow_tilt: float = 0.0    # + inner ends up (sad/worried), - inner down (angry/focused)
    brow_raise: float = 0.0   # -1..1
    mouth_curve: float = 0.1  # -1 frown .. 1 smile
    mouth_open: float = 0.0
    mouth_width: float = 0.5
    pupil_x: float = 0.0      # gaze -1..1
    pupil_y: float = 0.0
    blush: float = 0.0

    def lerp(self, other: FaceParams, k: float) -> FaceParams:
        return FaceParams(**{f.name: getattr(self, f.name) + (getattr(other, f.name) - getattr(self, f.name)) * k
                             for f in fields(self)})


NEUTRAL = FaceParams()

PRESETS: dict[str, FaceParams] = {
    "neutral": NEUTRAL,
    "happy": FaceParams(eye_open=0.7, eye_happy=0.6, brow_raise=0.2, mouth_curve=0.9, mouth_width=0.7, blush=0.4),
    "sad": FaceParams(eye_open=0.6, brow_tilt=0.8, brow_raise=-0.1, mouth_curve=-0.7, mouth_width=0.4, pupil_y=0.3),
    "surprised": FaceParams(eye_open=1.0, eye_scale=1.25, brow_raise=0.9, mouth_curve=0.0, mouth_open=0.7,
                            mouth_width=0.35),
    "empathy": FaceParams(eye_open=0.75, eye_happy=0.2, brow_tilt=0.5, mouth_curve=0.3, mouth_width=0.45, blush=0.2),
    "thinking": FaceParams(eye_open=0.7, pupil_x=0.6, pupil_y=-0.6, brow_tilt=-0.2, brow_raise=0.3,
                           mouth_curve=-0.1, mouth_width=0.3),
    "curious": FaceParams(eye_open=1.0, eye_scale=1.1, brow_raise=0.5, mouth_curve=0.3, mouth_open=0.15),
    "concerned": FaceParams(eye_open=0.85, brow_tilt=0.6, brow_raise=0.2, mouth_curve=-0.3, mouth_width=0.4),
    "excited": FaceParams(eye_open=1.0, eye_happy=0.5, eye_scale=1.15, brow_raise=0.6, mouth_curve=1.0,
                          mouth_open=0.5, mouth_width=0.8, blush=0.6),
    "calm": FaceParams(eye_open=0.5, eye_happy=0.3, mouth_curve=0.4, mouth_width=0.45),
    # internal states used by the micro-loop (not given to the LLM)
    "listening": FaceParams(eye_open=1.0, brow_raise=0.25, mouth_curve=0.2),
    "sleepy": FaceParams(eye_open=0.15, mouth_curve=0.0, mouth_width=0.3),
}


def target_params(name: str, intensity: float) -> FaceParams:
    preset = PRESETS.get(name, NEUTRAL)
    return NEUTRAL.lerp(preset, max(0.0, min(1.0, intensity)))


class FaceController:
    """Thread-safe state shared between dialogue/audio threads and the renderer (main thread)."""

    def __init__(self, default: str = "neutral"):
        self._lock = threading.Lock()
        self.name, self.intensity = default, 0.5
        self.target = target_params(default, 0.5)
        self.level = 0.0          # TTS loudness for lip-sync
        self.caption = ""
        self.debug = ""

    def set_expression(self, name: str, intensity: float = 0.6) -> None:
        with self._lock:
            self.name, self.intensity = name, intensity
            self.target = target_params(name, intensity)

    def set_level(self, rms: float) -> None:
        self.level = rms

    def set_caption(self, text: str) -> None:
        self.caption = text

    def snapshot(self) -> tuple[FaceParams, float]:
        with self._lock:
            return replace(self.target), self.level
