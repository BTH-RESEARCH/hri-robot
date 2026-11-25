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
    pad = np.zeros(int(0.2 * sample_rate), np.float32)   # the model misses words at the very edge
    s.accept_waveform(sample_rate, np.concatenate([pad, samples, pad]))
    recognizer.decode_stream(s)
    return s.result.text.strip().lower()   # model outputs UPPERCASE


class Segmenter:
    """VAD with pre-roll: silero marks the onset late for soft consonants, so every utterance also gets the
    `pre_roll` seconds before it (never audio of the previous utterance, never audio from while the robot spoke)."""

    KEEP_SEC = 25

    def __init__(self, vad, sr: int, pre_roll: float = 0.4, guard: float = 0.15, window: int = 512):
        self.vad, self.sr, self.window = vad, sr, window
        self.pre, self.guard = int(pre_roll * sr), int(guard * sr)
        self._restart()
        self._skip = 0

    def _restart(self) -> None:
        self.vad.reset()                          # segment positions count from here
        self.buf = np.zeros(0, np.float32)
        self.hist = np.zeros(0, np.float32)
        self.base = 0                             # position of hist[0]
        self.prev_end = 0
        self.in_speech = False

    def feed(self, chunk: np.ndarray, muted: bool) -> list[tuple[str, np.ndarray | None]]:
        if muted:                                 # the robot's turn: do not transcribe it
            if self._skip == 0 or len(self.hist) or len(self.buf):
                self._restart()
            self._skip = self.guard               # room echo right after the turn
            return []
        if self._skip > 0:
            n = min(self._skip, len(chunk))
            self._skip -= n
            chunk = chunk[n:]
        out: list[tuple[str, np.ndarray | None]] = []
        self.buf = np.concatenate([self.buf, chunk])
        while len(self.buf) >= self.window:
            w, self.buf = self.buf[: self.window], self.buf[self.window:]
            self.hist = np.concatenate([self.hist, w])
            extra = len(self.hist) - self.KEEP_SEC * self.sr
            if extra > 0:
                self.hist, self.base = self.hist[extra:], self.base + extra
            self.vad.accept_waveform(w)
            if self.vad.is_speech_detected() and not self.in_speech:
                self.in_speech = True
                out.append(("start", None))
            while not self.vad.empty():
                start, body = int(self.vad.front.start), np.array(self.vad.front.samples, np.float32)
                self.vad.pop()
                a = max(start - self.pre, self.prev_end, self.base)
                pre = self.hist[a - self.base: max(a, start) - self.base]
                self.prev_end = start + len(body)
                self.in_speech = False
                out.append(("end", np.concatenate([pre, body])))
        return out


def save_wav(folder, samples: np.ndarray, sr: int, text: str) -> None:
    import wave
    from pathlib import Path
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    log.info("saved %s  (%s)", f, text)


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
        v = acfg["vad"]
        self.pre_roll, self.guard = float(v.get("pre_roll_sec", 0.4)), float(v.get("guard_sec", 0.15))
        self.save_dir = None          # hri check asr --save DIR
        self.on_start, self.on_end = on_speech_start, on_speech_end
        self.is_muted = is_muted or (lambda: False)
        self._quit = threading.Event()
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def stop(self):
        self._quit.set()

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("mic status: %s", status)
        self._audio_q.put(indata[:, 0].copy())

    def run(self):
        try:
            self._listen()
        except Exception as e:  # noqa: BLE001 - e.g. no microphone plugged in: say so, keep the robot running
            log.error("Microphone unavailable, the robot cannot hear: %s\n"
                      "  Check that a mic is listed by:  arecord -l    then restart (hri check mic)", e)

    def _listen(self):
        seg = Segmenter(self.vad, self.sr, self.pre_roll, self.guard, self.WINDOW)
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=self.WINDOW,
                            device=self.device, callback=self._callback):
            log.info("ASR listening (sr=%d)", self.sr)
            while not self._quit.is_set():
                try:
                    chunk = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                for event, samples in seg.feed(chunk, self.is_muted()):
                    if event == "start":
                        if self.on_start:
                            self.on_start(time.time())
                        continue
                    t_end = time.time()
                    t_start = t_end - len(samples) / self.sr
                    if self.on_end:
                        self.on_end(t_start, t_end)
                    t0 = time.perf_counter()
                    text = transcribe(self.recognizer, samples, self.sr)
                    log.info("ASR (%.0f ms): %s", (time.perf_counter() - t0) * 1000, text)
                    if self.save_dir:
                        save_wav(self.save_dir, samples, self.sr, text)
                    if text:
                        self.out_q.put(Utterance(text, t_start, t_end))
