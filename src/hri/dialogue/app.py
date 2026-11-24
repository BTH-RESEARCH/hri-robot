"""Orchestrator: wires sensing -> affect engine -> intervention logic -> LLM -> TTS + face.

Threads / processes:
  main thread     : pygame face renderer (or idle wait when UI disabled)
  hri-vision (P)  : camera + YuNet + ResNet-18 FER -> vision queue (raw probs)
  vision-pump     : vision queue -> FERStream (EMA + personal baseline) -> AffectAggregator
  micro-loop      : every 1 s, V/A of the last 1 s -> Mirroring / Support / De-escalation face
  hri-asr         : mic + VAD + ASR             -> utterance queue   (or stdin in text mode)
  dialogue        : utterance -> speaker ID -> affect summary -> risk -> LLM stream -> segment queue
  speech          : segment queue -> face expression + TTS playback + lip-sync
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

from hri.affect.aggregator import AffectAggregator, congruence
from hri.affect.risk import assess_risk
from hri.affect.text_sentiment import text_valence
from hri.audio.types import Utterance
from hri.config import Config
from hri.dialogue.micro_loop import decide
from hri.llm.expression_parser import ExpressionStreamParser, Segment
from hri.llm.prompts import (
    build_history_message,
    build_system_prompt,
    build_user_message,
    guidance_after_intervention,
)
from hri.models import ModelRegistry
from hri.ui.expressions import FaceController
from hri.vision.fer import CIRCUMPLEX, FERPLUS_CLASSES, load_meta
from hri.vision.postproc import FERStream

log = logging.getLogger(__name__)

END_OF_TURN = None
GREETINGS = {  # Initial Engagement Protocol (cold start), chosen from the first seconds of FER
    "positive": Segment("happy", 0.8, "Xin chào! Trông bạn vui quá, hôm nay có chuyện gì hay không?"),
    "neutral": Segment("happy", 0.5, "Xin chào bạn, mình là Grace. Hôm nay bạn thế nào?"),
    "negative": Segment("empathy", 0.6, "Chào bạn. Mình ở đây nè, bạn có muốn trò chuyện một chút không?"),
}


def fake_affect(label: str) -> dict:
    v, a = CIRCUMPLEX.get(label, (0.0, 0.0))
    return {"face_visible": True, "dominant": label, "valence": v, "arousal": a,
            "trend": "stable", "stability": 0.9, "confidence": 0.8}


class RobotApp:
    def __init__(self, cfg: Config, text_mode: bool = False, use_vision: bool = True,
                 use_tts: bool = True, use_ui: bool = True, fake_emotion: str | None = None,
                 greet: bool = True):
        from hri.llm.client import LLMClient

        self.cfg = cfg
        self.reg = ModelRegistry(cfg)
        self.text_mode = text_mode
        self.use_ui = use_ui and cfg.get("ui.enabled", True)
        self.fake_emotion = fake_emotion
        self.greet_enabled = greet
        self.stop = threading.Event()
        self.utt_q: queue.Queue[Utterance] = queue.Queue()
        self.say_q: queue.Queue[Segment | None] = queue.Queue()
        self.state = "idle"   # calibrating | idle | listening | thinking | speaking

        self.allowed = cfg.get("expression.allowed")
        self.face = FaceController(cfg.get("expression.default", "neutral"))
        use_vision = use_vision and cfg.get("vision.enabled", True) and not fake_emotion
        # class names come from the FER pack (fer.json) -> fails early with "hri models download fer"
        self.labels = load_meta(self.reg.require(cfg.get("vision.fer.model")))["classes"] if use_vision \
            else list(FERPLUS_CLASSES)
        self.aggregator = AffectAggregator(cfg.section("affect"), self.labels)
        self.fer_stream = FERStream(self.labels, cfg.section("vision"))
        self._stream_lock = threading.Lock()
        self.vision_ready = threading.Event()
        self.last_micro: dict | None = None
        self.last_micro_t = 0.0
        self.llm = LLMClient(cfg.section("llm"))
        self.histories: dict[str, list[dict]] = defaultdict(list)
        self.pending_guidance: dict[str, str] = {}   # speaker -> Response Guidance for the next LLM turn

        self.speaker = None
        if use_tts:
            from hri.audio.tts import Speaker
            self.speaker = Speaker(self.reg, cfg.section("audio"))
            self.speaker.precache([g.text for g in GREETINGS.values()])

        self.vision = None
        if use_vision:
            from hri.vision.worker import VisionProcess
            self.vision = VisionProcess(cfg)

        self.sid = None
        self.asr = None
        if not text_mode:
            from hri.audio.asr import ASRWorker
            if cfg.get("audio.speaker_id.enabled", True):
                from hri.audio.speaker_id import SpeakerID
                self.sid = SpeakerID(self.reg, cfg.section("audio.speaker_id"), cfg.data_dir,
                                     int(cfg.get("audio.sample_rate", 16000)))
            self.asr = ASRWorker(cfg, self.reg, self.utt_q, self._on_speech_start, self._on_speech_end,
                                 self._asr_muted)

        log_dir = cfg.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.turn_log: Path = log_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    # ------------------------------------------------------------------ lifecycle
    def run(self) -> None:
        threads = [threading.Thread(target=self._dialogue_loop, name="dialogue", daemon=True),
                   threading.Thread(target=self._speech_loop, name="speech", daemon=True)]
        if self.vision:
            self.vision.start()
            threads.append(threading.Thread(target=self._vision_pump, name="vision-pump", daemon=True))
            if self.cfg.get("micro_loop.enabled", True):
                threads.append(threading.Thread(target=self._micro_loop, name="micro-loop", daemon=True))
        if self.asr:
            self.asr.start()
        if self.text_mode:
            threads.append(threading.Thread(target=self._stdin_loop, name="stdin", daemon=True))
        for t in threads:
            t.start()
        log.info("Robot running (profile=%s). Turn log: %s", self.cfg.profile, self.turn_log)
        try:
            if self.use_ui:
                from hri.ui.face_renderer import FaceRenderer
                renderer = FaceRenderer(self.cfg.section("ui"), self.face)
                renderer.on_key = self._on_key
                renderer.run(self.stop)
            else:
                while not self.stop.is_set():
                    self.stop.wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.stop.set()
        if self.speaker:
            self.speaker.stop()
        if self.asr:
            self.asr.stop()
        if self.vision:
            self.vision.stop()

    # ------------------------------------------------------------------ sensing callbacks
    def _asr_muted(self) -> bool:
        barge = self.cfg.get("audio.barge_in", False)
        return bool(self.speaker and self.speaker.playing.is_set() and not barge)

    def _on_speech_start(self, t: float) -> None:
        if self.speaker and self.speaker.playing.is_set() and self.cfg.get("audio.barge_in", False):
            self.speaker.stop()           # barge-in: user interrupts the robot
            self._drain(self.say_q)
        self.state = "listening"
        self.face.set_expression("listening", 0.7)

    def _on_speech_end(self, t0: float, t1: float) -> None:
        self.aggregator.mark_speaking(t0, t1)
        self.state = "thinking"
        self.face.set_expression("thinking", 0.6)   # instant feedback while ASR + LLM run

    def _on_key(self, ch: str) -> None:
        if ch == "c":                     # same shortcut as the FER demo: re-measure the baseline
            self.recalibrate()

    def recalibrate(self) -> None:
        with self._stream_lock:
            self.fer_stream.start_calibration()
        log.info("Baseline: re-calibrating (%.1fs of resting face)", self.fer_stream.calibrator.calib_seconds)

    def _vision_pump(self) -> None:
        while not self.stop.is_set():
            try:
                msg = self.vision.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg["type"] == "ready":
                log.info("Vision ready: %s", msg.get("model"))
                self.vision_ready.set()
                if msg.get("labels") and list(msg["labels"]) != list(self.labels):
                    log.error("FER labels mismatch: process %s vs pack %s", msg["labels"], self.labels)
                continue
            with self._stream_lock:
                frame = self.fer_stream.update(msg)
                calibrating = self.fer_stream.calibrating
            if frame is None:
                continue
            # frames measured for the baseline are the user's resting face, not affect evidence
            if not calibrating:
                self.aggregator.add(frame.to_msg(self.labels))
            self.face.debug = (f"user:{frame.label} {frame.confidence:.2f} V{frame.valence:+.2f} "
                               f"A{frame.arousal:+.2f}" + (" [calib]" if calibrating else ""))

    def _micro_loop(self) -> None:
        """Fluid-Xpress Micro-Loop: re-evaluate the robot face every update_sec from the last window_sec."""
        mcfg = self.cfg.section("micro_loop")
        period, window = float(mcfg.get("update_sec", 1.0)), float(mcfg.get("window_sec", 1.0))
        while not self.stop.wait(period):
            # active while the user talks and during LLM latency; the LLM's [EMO] tags own the face
            # while the robot speaks; paused during calibration / cold start
            if self.state not in ("idle", "listening", "thinking") or self.fer_stream.calibrating:
                continue
            now = time.time()
            d = decide(self.aggregator.frames_between(now - window, now), self.labels, mcfg)
            if d is None:
                continue
            self.face.set_expression(d.expression, d.intensity)
            if self.last_micro is None or self.last_micro["strategy"] != d.strategy:
                log.info("Micro-loop: %s -> %s (V%+.2f A%+.2f, user %s)", d.strategy, d.expression,
                         d.valence, d.arousal, d.dominant)
            self.last_micro, self.last_micro_t = d.to_dict(), now

    def _stdin_loop(self) -> None:
        # Windows: a thread blocked in input() on a *piped* stdin (Git Bash, scripts) makes the spawned
        # vision process hang while Python initialises its stdio -> only read stdin once vision is up.
        if self.vision:
            self.vision_ready.wait(60)
        print("Text mode: type a message and press Enter (Ctrl+C / 'exit' to quit)")
        while not self.stop.is_set():
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.stop.set()
                return
            if text.lower() in ("exit", "quit"):
                self.stop.set()
                return
            if text:
                now = time.time()
                self.utt_q.put(Utterance(text, now - max(1.0, len(text) / 15), now))

    # ------------------------------------------------------------------ macro-loop
    def _dialogue_loop(self) -> None:
        if self.greet_enabled:
            self._greet()
        while not self.stop.is_set():
            try:
                utt = self.utt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handle(utt)
            except Exception:
                log.exception("turn failed")
                self.say_q.put(Segment("concerned", 0.5, "Xin lỗi, mình đang gặp chút trục trặc, bạn nói lại được không?"))
                self.say_q.put(END_OF_TURN)

    def _greet(self) -> None:
        """Initial Engagement Protocol: wait for a face -> measure the resting-face baseline (3 s)
        -> classify the next N calibrated frames as positive / neutral / negative -> greet."""
        ccfg = self.cfg.section("cold_start")
        v = 0.0
        if self.fake_emotion:
            v = fake_affect(self.fake_emotion)["valence"]
        elif self.vision:
            self.state = "calibrating"
            self.face.set_expression("listening", 0.6)
            log.info("Cold start: waiting for a face...")
            while not self.stop.is_set() and not self._face_seen():
                time.sleep(0.1)
            self.recalibrate()
            while not self.stop.is_set() and self.fer_stream.calibrating:
                time.sleep(0.1)
            t_ready = time.time()
            log.info("Cold start: baseline %s", self.fer_stream.calibrator.info())
            n = int(ccfg.get("frames", 12))
            frames: list = []
            while not self.stop.is_set() and len(frames) < n:
                frames = self.aggregator.frames_between(t_ready, time.time())
                time.sleep(0.05)
            frames = frames[:n]
            if frames:
                q = [max(1e-3, f.quality) for f in frames]
                v = sum(f.valence * w for f, w in zip(frames, q)) / sum(q)
        if self.stop.is_set():
            return
        key = ("positive" if v >= float(ccfg.get("positive_valence", 0.2)) else
               "negative" if v <= float(ccfg.get("negative_valence", -0.2)) else "neutral")
        log.info("Cold start: mean V %+.2f -> %s greeting", v, key)
        self.state = "idle"
        self.say_q.put(GREETINGS[key])
        self.say_q.put(END_OF_TURN)

    def _face_seen(self) -> bool:
        return self.fer_stream.face_seen

    def _handle(self, utt: Utterance) -> None:
        t_start = time.time()
        speaker = "user"
        if self.sid is not None and utt.samples is not None:
            speaker = self.sid.identify(utt.samples) or "user"
        utt.speaker = speaker

        affect = fake_affect(self.fake_emotion) if self.fake_emotion else \
            self.aggregator.summarize(utt.t_start, utt.t_end, speaker)
        if affect.get("face_visible"):
            affect["text_vs_face"] = congruence(affect.get("valence"), text_valence(utt.text),
                                                float(self.cfg.get("affect.incongruence_threshold", 0.6)))
        risk = assess_risk(utt.text, affect, self.cfg.section("safety"))
        affect["risk"] = risk
        self.face.set_caption(f"[{speaker}] {utt.text}")
        log.info("USER[%s]: %s | affect=%s", speaker, utt.text, affect)

        hist = self.histories[speaker]
        if risk == "high":   # Immediate Support: local, no LLM round-trip
            seg = Segment("concerned", 0.8, self.cfg.get("safety.support_message"))
            self.say_q.put(seg)
            self.say_q.put(END_OF_TURN)
            # keep the LLM in the loop next turn: history + Response Guidance (Fluid-Xpress Sec. 3.3)
            hist += [{"role": "user", "content": build_history_message(utt.text, affect)},
                     {"role": "assistant", "content": f"[EMO:{seg.emotion}|{seg.intensity}] {seg.text}"}]
            self.pending_guidance[speaker] = guidance_after_intervention(seg.text)
            self._log_turn(utt, affect, seg.text, t_start, None, intervention="immediate_support")
            return

        # what the micro-loop is showing right now, so the first [EMO] tag continues smoothly from it
        robot_face = self.last_micro if self.last_micro and time.time() - self.last_micro_t < 3.0 else None
        guidance = self.pending_guidance.pop(speaker, None)
        n = int(self.cfg.get("llm.history_turns", 8)) * 2
        user_msg = build_user_message(utt.text, affect, robot_face, guidance)
        messages = [{"role": "system", "content": build_system_prompt(self.allowed, user_name=speaker)},
                    *hist[-n:], {"role": "user", "content": user_msg}]
        log.debug("LLM input:\n%s", user_msg)

        parser = ExpressionStreamParser(self.allowed, self.cfg.get("expression.default", "neutral"))
        full, t_first = "", None
        for delta in self.llm.stream(messages):
            if self.stop.is_set():
                return
            full += delta
            for seg in parser.feed(delta):
                t_first = t_first or time.time()
                self.say_q.put(seg)
        for seg in parser.flush():
            t_first = t_first or time.time()
            self.say_q.put(seg)
        self.say_q.put(END_OF_TURN)

        # compact history: the full AFFECT JSON is only sent for the current turn
        hist += [{"role": "user", "content": build_history_message(utt.text, affect)},
                 {"role": "assistant", "content": full}]
        self._log_turn(utt, affect, full, t_start, t_first, robot_face=robot_face, guidance=guidance)

    def _speech_loop(self) -> None:
        while not self.stop.is_set():
            try:
                seg = self.say_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if seg is END_OF_TURN:
                self.state = "idle"
                continue
            self.state = "speaking"
            self.face.set_expression(seg.emotion, seg.intensity)
            print(f"ROBOT [{seg.emotion}|{seg.intensity:.1f}]: {seg.text}", flush=True)
            if self.speaker:
                self.speaker.say(seg.text, on_level=self.face.set_level)
            else:
                time.sleep(min(4.0, 0.06 * len(seg.text)))   # simulate speaking time

    # ------------------------------------------------------------------ research log
    def _log_turn(self, utt: Utterance, affect: dict, reply: str, t_start: float, t_first: float | None,
                  robot_face: dict | None = None, guidance: str | None = None, intervention: str | None = None):
        rec = {
            "t": utt.t_end, "speaker": utt.speaker, "text": utt.text, "affect": affect, "reply": reply,
            "micro_loop": self.last_micro, "robot_face_sent": robot_face, "guidance": guidance,
            "intervention": intervention,
            "latency": {
                "speech_end_to_handle": round(t_start - utt.t_end, 3),
                "speech_end_to_first_segment": round(t_first - utt.t_end, 3) if t_first else None,
            },
        }
        with open(self.turn_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
