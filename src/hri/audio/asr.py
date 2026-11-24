"""Mic -> Silero VAD -> Zipformer (offline, per utterance) with sherpa-onnx.

The Vietnamese zipformer is a non-streaming model: VAD cuts each utterance and we decode it as
soon as the user stops talking (usually < 300 ms on Pi5 for a short sentence).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from hri.audio.types import Utterance
from hri.models import ModelRegistry, find_file

log = logging.getLogger(__name__)


def build_recognizer(reg: ModelRegistry, asr_cfg: dict):
    import sherpa_onnx

    d = reg.require(asr_cfg["model"])
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(find_file(d, "encoder*.onnx")),
        decoder=str(find_file(d, "decoder*.onnx")),
        joiner=str(find_file(d, "joiner*.onnx")),
        tokens=str(find_file(d, "tokens.txt")),
        num_threads=int(asr_cfg.get("num_threads", 2)),
        decoding_method=asr_cfg.get("decoding_method", "greedy_search"),
    )


def build_vad(reg: ModelRegistry, vad_cfg: dict, sample_rate: int):
    import sherpa_onnx

    c = sherpa_onnx.VadModelConfig()
    c.silero_vad.model = str(reg.require(vad_cfg["model"]))
    c.silero_vad.threshold = float(vad_cfg.get("threshold", 0.5))
    c.silero_vad.min_silence_duration = float(vad_cfg.get("min_silence_duration", 0.5))
    c.silero_vad.min_speech_duration = float(vad_cfg.get("min_speech_duration", 0.25))
    c.silero_vad.max_speech_duration = float(vad_cfg.get("max_speech_duration", 15))
    c.sample_rate = sample_rate
    return sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=30)


def transcribe(recognizer, samples: np.ndarray, sample_rate: int) -> str:
    s = recognizer.create_stream()
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(s)
    return s.result.text.strip().lower()   # model outputs UPPERCASE


class ASRWorker(threading.Thread):
    """Emits Utterance objects on `out_q`. Callbacks let the app react instantly to VAD events
    (micro-loop: switch to 'listening'/'thinking' face before ASR text exists)."""

    WINDOW = 512  # silero frame size at 16 kHz

    def __init__(self, cfg, reg: ModelRegistry, out_q: queue.Queue,
                 on_speech_start: Callable[[float], None] | None = None,
                 on_speech_end: Callable[[float, float], None] | None = None,
                 is_muted: Callable[[], bool] | None = None):
        super().__init__(name="hri-asr", daemon=True)
        acfg = cfg.section("audio")
        self.sr = int(acfg.get("sample_rate", 16000))
        self.device = acfg.get("input_device")
        self.recognizer = build_recognizer(reg, acfg["asr"])
        self.vad = build_vad(reg, acfg["vad"], self.sr)
        self.out_q = out_q
        self.on_start, self.on_end = on_speech_start, on_speech_end
        self.is_muted = is_muted or (lambda: False)
        self._stop = threading.Event()
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def stop(self):
        self._stop.set()

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("mic status: %s", status)
        self._audio_q.put(indata[:, 0].copy())

    def run(self):
        buf = np.zeros(0, np.float32)
        in_speech, speech_t0 = False, 0.0
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=self.WINDOW,
                            device=self.device, callback=self._callback):
            log.info("ASR listening (sr=%d)", self.sr)
            while not self._stop.is_set():
                try:
                    chunk = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.is_muted():          # robot is speaking and barge-in disabled
                    buf = np.zeros(0, np.float32)
                    self.vad.reset()
                    in_speech = False
                    continue
                buf = np.concatenate([buf, chunk])
                while len(buf) >= self.WINDOW:
                    self.vad.accept_waveform(buf[: self.WINDOW])
                    buf = buf[self.WINDOW:]
                    if self.vad.is_speech_detected() and not in_speech:
                        in_speech, speech_t0 = True, time.time()
                        if self.on_start:
                            self.on_start(speech_t0)
                while not self.vad.empty():
                    seg = np.array(self.vad.front.samples, dtype=np.float32)
                    self.vad.pop()
                    t_end = time.time()
                    t_start = t_end - len(seg) / self.sr
                    in_speech = False
                    if self.on_end:
                        self.on_end(t_start, t_end)
                    t0 = time.perf_counter()
                    text = transcribe(self.recognizer, seg, self.sr)
                    log.info("ASR (%.0f ms): %s", (time.perf_counter() - t0) * 1000, text)
                    if text:
                        self.out_q.put(Utterance(text, t_start, t_end, seg))
