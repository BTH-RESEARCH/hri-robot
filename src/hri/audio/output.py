"""Open the loudspeaker, including a Raspberry Pi HDMI screen's speakers, and play without losing the end.

vc4-hdmi accepts a single client at a time and only a few formats, so: nothing else in this process may open
audio (see ui/display.py: pygame's mixer is never initialised), and we try the device's converting ALSA
aliases and, if needed, 48 / 44.1 kHz stereo instead of the voice's 22.05 kHz mono.

HDMI and Bluetooth sinks buffer sound after the driver says it is done: closing the stream after each sentence
cuts its last word. `Player` therefore keeps one stream open for the whole turn (feeding silence between
sentences) and adds silence before closing.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable

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

    def open(self, sr: int, callback=None):
        """-> (OutputStream, rate, channels). Raises RuntimeError with every reason."""
        combos = [(d, r, c) for d in self.devices for r, c in ((sr, 1), (48000, 2), (44100, 2))]
        if self.good:
            combos = [self.good, *[c for c in combos if c != self.good]]
        errors = []
        for dev, rate, ch in combos:
            try:
                kw = {"callback": callback} if callback else {}
                stream = self.sd.OutputStream(samplerate=rate, channels=ch, dtype="float32", device=dev, **kw)
            except Exception as e:  # noqa: BLE001 - PortAudioError, ValueError (unknown name): try the next one
                errors.append(f"{dev or 'system default'} {rate} Hz x{ch}: {e}")
                continue
            if self.good != (dev, rate, ch):
                self.good = (dev, rate, ch)
                log.info("Speaker: %s, %d Hz, %d channel(s)", dev or "system default", rate, ch)
            return stream, rate, ch
        raise RuntimeError(HELP + "\n  " + "\n  ".join(errors))


class Player:
    """One open stream per turn. play() blocks until the sentence has been handed to the device."""

    def __init__(self, output: Output, lead_sec: float = 0.15, tail_sec: float = 0.5, gap_sec: float = 0.12):
        self.output = output
        self.lead, self.tail, self.gap = lead_sec, tail_sec, gap_sec
        self._stream = None
        self._rate = 0
        self._lock = threading.Lock()
        self._buf = np.zeros(0, np.float32)
        self._done = threading.Event()
        self._interrupt = threading.Event()
        self._level: Callable[[float], None] | None = None

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def _callback(self, out, frames, time_info, status) -> None:
        if status:
            log.debug("speaker: %s", status)
        with self._lock:
            x, self._buf = self._buf[:frames], self._buf[frames:]
            empty = not len(self._buf)
        out[:] = 0
        out[: len(x)] = x[:, None]
        if self._level:
            self._level(float(min(1.0, np.sqrt(np.mean(x ** 2)) * 6)) if len(x) else 0.0)
        if empty:
            self._done.set()

    def _open(self, sr: int) -> bool:
        """True if the stream had to be opened now."""
        if self._stream is not None:
            return False
        self._stream, self._rate, _ = self.output.open(sr, self._callback)
        self._stream.start()
        return True

    def play(self, samples: np.ndarray, sr: int, on_level: Callable[[float], None] | None = None) -> bool:
        """False if interrupted. Raises if the speaker cannot be opened or stalls."""
        self._interrupt.clear()
        fresh = self._open(sr)
        data = convert(samples, sr, self._rate, 1)[:, 0]
        data = np.concatenate([np.zeros(int(self._rate * (self.lead if fresh else 0.0)), np.float32), data,
                               np.zeros(int(self._rate * self.gap), np.float32)])
        self._level = on_level
        with self._lock:
            self._buf = data
            self._done.clear()
        deadline = time.monotonic() + len(data) / self._rate + 5.0
        try:
            while not self._done.wait(0.02):
                if self._interrupt.is_set():
                    with self._lock:
                        self._buf = np.zeros(0, np.float32)
                    return False
                if time.monotonic() > deadline:
                    self.close(wait=False)
                    raise RuntimeError("speaker stalled")
            return True
        finally:
            self._level = None
            if on_level:
                on_level(0.0)

    def close(self, wait: bool = True) -> None:
        """End of the turn: let the device play everything it has buffered, then release it."""
        s, self._stream = self._stream, None
        if s is None:
            return
        try:
            if wait:
                latency = s.latency if isinstance(getattr(s, "latency", 0), int | float) else 0
                self._interrupt.wait(self.tail + latency)      # the callback keeps feeding silence meanwhile
            s.stop()
        finally:
            s.close()

    def interrupt(self) -> None:
        self._interrupt.set()
