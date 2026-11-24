"""Open the loudspeaker, including a Raspberry Pi HDMI screen's speakers.

vc4-hdmi accepts a single client at a time and only a few formats, so: nothing else in this process may open
audio (see ui/display.py: pygame's mixer is never initialised), and we try the device's converting ALSA
aliases and, if needed, 48 / 44.1 kHz stereo instead of the voice's 22.05 kHz mono.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

log = logging.getLogger(__name__)

HELP = """No loudspeaker could be opened.
  - 'Device unavailable' = another program is using it:  sudo fuser -v /dev/snd/*
  - list devices:  aplay -l && python -m sounddevice      then test:  hri check speaker
  - HDMI screen: plug it into HDMI0 before boot, volume not muted; or choose a device:
      hri run --set audio.output_device=sysdefault"""


def convert(samples: np.ndarray, sr: int, rate: int, channels: int) -> np.ndarray:
    """float32 mono -> (n, channels) at `rate`."""
    x = np.asarray(samples, np.float32).reshape(-1)
    if rate != sr and len(x):
        n = max(1, round(len(x) * rate / sr))
        x = np.interp(np.arange(n) * (sr / rate), np.arange(len(x)), x).astype(np.float32)
    return np.repeat(x[:, None], channels, axis=1)


class Output:
    """Remembers the first (device, rate, channels) that works."""

    def __init__(self, device=None, sd=None, platform: str = sys.platform):
        if sd is None:
            import sounddevice as sd
        self.sd = sd
        self.devices = [device]
        if platform.startswith("linux"):
            self.devices += [d for d in (None, "default", "sysdefault", "hdmi") if d not in self.devices]
        self.good: tuple | None = None

    def open(self, sr: int):
        """-> (started-on-enter OutputStream, rate, channels). Raises RuntimeError with every reason."""
        combos = [(d, r, c) for d in self.devices for r, c in ((sr, 1), (48000, 2), (44100, 2))]
        if self.good:
            combos = [self.good, *[c for c in combos if c != self.good]]
        errors = []
        for dev, rate, ch in combos:
            try:
                stream = self.sd.OutputStream(samplerate=rate, channels=ch, dtype="float32", device=dev)
            except Exception as e:  # noqa: BLE001 - PortAudioError, ValueError (unknown name): try the next one
                errors.append(f"{dev or 'system default'} {rate} Hz x{ch}: {e}")
                continue
            if self.good != (dev, rate, ch):
                self.good = (dev, rate, ch)
                log.info("Speaker: %s, %d Hz, %d channel(s)", dev or "system default", rate, ch)
            return stream, rate, ch
        raise RuntimeError(HELP + "\n  " + "\n  ".join(errors))
