"""Robot face = two glowing eyes (ported from faces/index.html). Every expression is a point in parameter space:
intensity blends from neutral, the renderer tweens between expressions."""
from __future__ import annotations

import threading
from dataclasses import dataclass, fields, replace


@dataclass
class FaceParams:
    w: float = 1.0        # eye width / height factors
    h: float = 1.0
    r: float = 1.0        # corner roundness
    top: float = 0.0      # upper / lower lid, 0..1 of the eye height
    bot: float = 0.0
    slant: float = 0.0    # upper lid angle in degrees: + = angry, - = sad
    hL: float = 1.0       # height of the left / right eye (asymmetry)
    hR: float = 1.0
    heart: float = 0.0    # 0 = eyes, 1 = hearts
    dx: float = 0.0       # gaze offset in px of the 800x480 design
    dy: float = 0.0
    tilt: float = 0.0     # whole face, degrees clockwise
    dim: float = 1.0      # brightness
    scale: float = 1.0

    def lerp(self, other: FaceParams, k: float) -> FaceParams:
        return FaceParams(**{f.name: getattr(self, f.name) + (getattr(other, f.name) - getattr(self, f.name)) * k
                             for f in fields(self)})


NEUTRAL = FaceParams()

# the ten presets of faces/index.html, unchanged
DESIGN: dict[str, FaceParams] = {
    "neutral": NEUTRAL,
    "happy": FaceParams(bot=0.40, h=1.02),
    "sad": FaceParams(top=0.30, slant=-17, h=0.95, dy=10),
    "angry": FaceParams(top=0.34, slant=19),
    "surprised": FaceParams(w=1.1, h=1.22),
    "sleepy": FaceParams(top=0.55, bot=0.10, dy=14),
    "suspicious": FaceParams(top=0.30, bot=0.26, w=1.06),
    "confused": FaceParams(hL=1.14, hR=0.72, tilt=-4, top=0.08),
    "love": FaceParams(heart=1),
    "sleep": FaceParams(top=0.94, dim=0.55),
}

# what the application shows. 'angry' and 'suspicious' are left out on purpose: the robot never looks hostile at a
# child; 'confused' is not used
PRESETS: dict[str, FaceParams] = {
    **{k: DESIGN[k] for k in ("neutral", "happy", "sad", "surprised", "love", "sleepy", "sleep")},
    "curious": FaceParams(scale=1.12),                                                # the neutral eyes, a little bigger
    "excited": FaceParams(bot=0.30, w=1.08, h=1.14, scale=1.04),                      # surprised + happy
    "empathy": FaceParams(top=0.16, slant=-11, bot=0.12, dy=6),                       # soft sad + a hint of a smile
    "concerned": FaceParams(top=0.22, slant=-15, hL=0.96, hR=0.96, tilt=-2, dy=4),
    "calm": FaceParams(top=0.24, bot=0.16, dy=6),                                     # relaxed lids
    "thinking": FaceParams(top=0.20, bot=0.06, dx=62, dy=-34, tilt=2),                # looks up and aside
    "listening": FaceParams(h=1.06, w=1.03),                                          # wide and attentive
}


def target_params(name: str, intensity: float) -> FaceParams:
    """The LLM mostly uses 0.4 - 0.8; spread that range so that expressions stay readable (0.8 = full design)."""
    i = max(0.0, min(1.0, intensity))
    return NEUTRAL.lerp(PRESETS.get(name, NEUTRAL), min(1.0, 0.3 + 0.875 * i) if i > 0 else 0.0)


class FaceController:
    """Thread-safe state shared between dialogue/audio threads and the renderer (main thread)."""

    def __init__(self, default: str = "neutral"):
        self._lock = threading.Lock()
        self.name, self.intensity = default, 0.5
        self.target = target_params(default, 0.5)
        self.level = 0.0          # TTS loudness: the eyes pulse while the robot speaks
        self.caption = ""
        self.mic_open = False     # green dot: the robot is listening
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
