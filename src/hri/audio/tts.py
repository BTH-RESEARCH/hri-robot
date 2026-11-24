"""Text -> speech with sherpa-onnx VITS (Piper Vietnamese), interruptible playback + lip-sync RMS."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from hri.audio.output import Output, convert
from hri.models import ModelRegistry, find_file

log = logging.getLogger(__name__)


class PiperTTS:
    """sherpa-onnx >= 2.0 no longer bundles espeak-ng: Piper models need phonemes computed
    externally with `piper_phonemize`, then passed as GenerationConfig.phoneme_codepoints."""

    def __init__(self, reg: ModelRegistry, tts_cfg: dict):
        import sherpa_onnx
        from piper_phonemize import phonemize_espeak

        d = reg.require(tts_cfg["model"])
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(find_file(d, "*.onnx")), tokens=str(find_file(d, "tokens.txt")))
        conf = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=int(tts_cfg.get("num_threads", 2))),
            max_num_sentences=1,
        )
        self._sherpa = sherpa_onnx
        self._phonemize = phonemize_espeak
        self.tts = sherpa_onnx.OfflineTts(conf)
        self.lang = tts_cfg.get("espeak_voice") or self.tts.lang or "vi"
        self.sample_rate = self.tts.sample_rate

    def generate(self, text: str, sid: int = 0, speed: float = 1.0) -> tuple[np.ndarray, int]:
        sentences = self._phonemize(text, self.lang)
        g = self._sherpa.GenerationConfig()
        g.sid = sid
        g.speed = speed
        g.phoneme_codepoints = [[ord(c) for c in s] for s in sentences]
        audio = self.tts.generate("", g)
        if not len(audio.samples):
            raise RuntimeError(f"TTS produced no audio for {text!r}")
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


def build_tts(reg: ModelRegistry, tts_cfg: dict) -> PiperTTS:
    return PiperTTS(reg, tts_cfg)


class Speaker:
    """Synthesizes and plays one sentence at a time. stop() interrupts immediately (barge-in)."""

    CHUNK = 1024

    def __init__(self, reg: ModelRegistry, audio_cfg: dict):
        tcfg = audio_cfg["tts"]
        self.tts = build_tts(reg, tcfg)
        self.sid = int(tcfg.get("speaker_id", 0))
        self.speed = float(tcfg.get("speed", 1.0))
        self.output = Output(audio_cfg.get("output_device"))
        self.playing = threading.Event()
        self._interrupt = threading.Event()
        self._cache: dict[str, tuple[np.ndarray, int]] = {}

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        if text in self._cache:
            return self._cache[text]
        t0 = time.perf_counter()
        out = self.tts.generate(text, sid=self.sid, speed=self.speed)
        log.debug("TTS %.0f ms for %r", (time.perf_counter() - t0) * 1000, text[:40])
        return out

    def precache(self, phrases: list[str]) -> None:
        """Pre-synthesize fillers ('Ừm...', greetings) for zero-latency playback."""
        for p in phrases:
            self._cache[p] = self.synth(p)

    def say(self, text: str, on_level: Callable[[float], None] | None = None) -> bool:
        """Blocking. Returns False if interrupted. on_level(rms 0..1) drives the mouth."""
        samples, sr = self.synth(text)
        self._interrupt.clear()
        self.playing.set()
        try:
            stream, rate, channels = self.output.open(sr)
            data = convert(samples, sr, rate, channels)
            with stream as out:
                for i in range(0, len(data), self.CHUNK):
                    if self._interrupt.is_set():
                        return False
                    chunk = data[i:i + self.CHUNK]
                    if on_level:
                        on_level(float(min(1.0, np.sqrt(np.mean(chunk ** 2)) * 6)))
                    out.write(chunk)
            return True
        finally:
            if on_level:
                on_level(0.0)
            self.playing.clear()

    def stop(self) -> None:
        self._interrupt.set()
