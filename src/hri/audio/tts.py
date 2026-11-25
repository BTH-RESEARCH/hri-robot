"""Text -> speech with sherpa-onnx VITS (Piper Vietnamese), interruptible playback + lip-sync RMS."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from hri.audio.output import Output, Player
from hri.models import ModelRegistry, find_file

log = logging.getLogger(__name__)


# Piper stops ~35 ms after the last vowel, so its end is easily lost; a pause phoneme after the final mark lets the
# voice die out (measured: energy of the last 30 ms 0.050 -> 0.005, tail 37 -> 141 ms). At text level it does not work.
END_PAUSE = " , "


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
        g.phoneme_codepoints = [[ord(c) for c in [*s, *END_PAUSE]] for s in sentences]
        audio = self.tts.generate("", g)
        if not len(audio.samples):
            raise RuntimeError(f"TTS produced no audio for {text!r}")
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


def build_tts(reg: ModelRegistry, tts_cfg: dict) -> PiperTTS:
    return PiperTTS(reg, tts_cfg)


class Speaker:
    """Synthesizes and plays one sentence at a time. stop() interrupts immediately (child distress)."""

    def __init__(self, reg: ModelRegistry, audio_cfg: dict):
        tcfg = audio_cfg["tts"]
        self.tts = build_tts(reg, tcfg)
        self.sid = int(tcfg.get("speaker_id", 0))
        self.speed = float(tcfg.get("speed", 1.0))
        self.player = Player(Output(audio_cfg.get("output_device")), tail_sec=float(tcfg.get("tail_sec", 0.5)))
        self.playing = threading.Event()
        self._cache: dict[tuple[str, float], tuple[np.ndarray, int]] = {}

    def synth(self, text: str, pace: float = 1.0) -> tuple[np.ndarray, int]:
        if (text, pace) in self._cache:
            return self._cache[text, pace]
        t0 = time.perf_counter()
        out = self.tts.generate(text, sid=self.sid, speed=self.speed * pace)
        log.debug("TTS %.0f ms for %r", (time.perf_counter() - t0) * 1000, text[:40])
        return out

    def precache(self, phrases: list[str], pace: float = 1.0) -> None:
        """Pre-synthesize fillers ('Ừm...', greetings) for zero-latency playback."""
        for p in phrases:
            self._cache[p, pace] = self.synth(p, pace)

    def say(self, text: str, on_level: Callable[[float], None] | None = None,
            pace: float = 1.0) -> bool:
        """Blocking. Returns False if interrupted. on_level(rms 0..1) drives the mouth.
        The loudspeaker stays open until end_turn(), so the last word of a sentence is never cut."""
        samples, sr = self.synth(text, pace)
        self.playing.set()
        return self.player.play(samples, sr, on_level)

    def end_turn(self) -> None:
        """Play everything out, then release the loudspeaker and let the microphone listen again."""
        try:
            self.player.close()
        finally:
            self.playing.clear()

    def stop(self) -> None:
        self.player.interrupt()
