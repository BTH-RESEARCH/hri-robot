"""Orchestrator: wires sensing -> affect engine -> intervention logic -> LLM -> TTS + face.

Threads / processes:
  main thread     : pygame face renderer (or idle wait when UI disabled)
  hri-vision (P)  : camera + YuNet + ResNet-18 FER -> vision queue (raw probs)
  vision-pump     : vision queue -> FERStream (sensitivity + baseline + hysteresis) -> AffectAggregator
  micro-loop      : every 1 s, V/A of the last 1 s -> Mirroring / Support / De-escalation face
  hri-asr         : mic + VAD + ASR             -> utterance queue   (or stdin in text mode)
  dialogue        : session start/end; utterance -> affect summary -> risk -> LLM stream -> segment queue
  speech          : segment queue -> face expression + TTS playback + lip-sync
  summary         : short-lived, folds old turns into the session summary after the robot has spoken
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from hri.affect.aggregator import AffectAggregator, congruence
from hri.affect.risk import assess_risk
from hri.affect.text_sentiment import text_valence
from hri.audio.types import Utterance
from hri.config import Config
from hri.dialogue.memory import SessionMemory, is_farewell
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
APOLOGY = "Xin lỗi, mình đang gặp chút trục trặc, bạn nói lại được không?"
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
        # memory of the current encounter only (RAM, wiped when the user leaves)
        self.memory = SessionMemory(cfg.section("memory"))
        self.active = False               # a session is running
        self.end_requested = False        # farewell or key 'n': end once the robot has finished speaking
        self.wait_absence = False         # said goodbye but still in view: do not greet again yet
        self.last_activity = time.time()  # last speech from the user or the robot
        self.last_face = 0.0

        self._last_audio_error = ""
        self.speaker = None
        if use_tts:
            from hri.audio.tts import Speaker
            self.speaker = Speaker(self.reg, cfg.section("audio"))
            self.speaker.precache([g.text for g in GREETINGS.values()])

        self.vision = None
        if use_vision:
            from hri.vision.worker import VisionProcess
            self.vision = VisionProcess(cfg)

        self.asr = None
        if not text_mode:
            from hri.audio.asr import ASRWorker
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
        self.last_activity = time.time()
        if self.speaker and self.speaker.playing.is_set() and self.cfg.get("audio.barge_in", False):
            self.speaker.stop()           # barge-in: user interrupts the robot
            self._drain(self.say_q)
        self.state = "listening"
        self.face.set_expression("listening", 0.7)

    def _on_speech_end(self, t0: float, t1: float) -> None:
        self.last_activity = time.time()
        self.aggregator.mark_speaking(t0, t1)
        self.state = "thinking"
        self.face.set_expression("thinking", 0.6)   # instant feedback while ASR + LLM run

    def _on_key(self, ch: str) -> None:
        if ch == "c":                     # same shortcut as the FER demo: re-measure the baseline
            self.recalibrate()
        elif ch == "n":                   # next user: forget this conversation, greet again
            self.end_requested, self.wait_absence = True, False

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
            self.last_face = time.time()
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
        # without a camera nobody "arrives": greet once at start-up
        if not self.vision and self.greet_enabled:
            self._start_session()
            self._greet()
        while not self.stop.is_set():
            if not self.active:
                self._wait_for_user()
                continue
            if self._session_over():
                self._end_session()
                continue
            try:
                utt = self.utt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handle(utt)
            except Exception:
                log.exception("turn failed")
                self.say_q.put(Segment("concerned", 0.5, APOLOGY))
                self.say_q.put(END_OF_TURN)

    # ------------------------------------------------------------------ session lifecycle
    def _face_present(self, within: float = 1.0) -> bool:
        return time.time() - self.last_face < within

    def _wait_for_user(self) -> None:
        """No session: somebody in front of the camera gets a greeting, somebody who just talks gets an answer."""
        if not self.utt_q.empty():
            self.wait_absence = False
            self._start_session()
            return
        if self.wait_absence and not self._face_present(float(self.cfg.get("memory.rearm_sec", 5))):
            self.wait_absence = False
        if self.vision and not self.wait_absence and self._face_present():
            self._start_session()
            if self.greet_enabled:
                self._greet()
            return
        self.stop.wait(0.1)

    def _start_session(self) -> None:
        self.active, self.end_requested = True, False
        self.last_activity = time.time()
        log.info("Session %d started", self.memory.session)   # memory is empty: wiped at start-up / session end

    def _session_over(self) -> bool:
        if self.state == "speaking" or not self.say_q.empty() or not self.utt_q.empty():
            return False
        if self.end_requested:
            return True
        quiet = time.time() - self.last_activity
        if not self.vision:
            return quiet > float(self.cfg.get("memory.idle_no_vision_sec", 180))
        idle = float(self.cfg.get("memory.idle_sec", 45))
        return quiet > idle and not self._face_present(idle)

    def _end_session(self) -> None:
        log.info("Session %d ended after %.0fs of silence -> memory wiped", self.memory.session,
                 time.time() - self.last_activity)
        self.memory.reset()
        self.active = self.end_requested = False
        self.aggregator.reset()               # frames and baseline of the previous person
        with self._stream_lock:
            self.fer_stream.cancel_calibration()
        self.last_micro = None
        self.state = "idle"
        self.face.set_caption("")
        self.face.set_expression(self.cfg.get("expression.default", "neutral"), 0.5)

    def _summarize(self) -> None:
        if self.memory.compact(lambda p: self.llm.complete([{"role": "user", "content": p}])):
            log.info("Memory: older turns summarised (%d chars)", len(self.memory.summary))

    def _greet(self) -> None:
        """Initial Engagement Protocol: measure the resting-face baseline (3 s) -> classify the next N
        calibrated frames as positive / neutral / negative -> greet. Gives up if the person walks away."""
        ccfg = self.cfg.section("cold_start")
        v = 0.0
        if self.fake_emotion:
            v = fake_affect(self.fake_emotion)["valence"]
        elif self.vision:
            self.state = "calibrating"
            self.face.set_expression("listening", 0.6)
            deadline = time.time() + float(ccfg.get("timeout_sec", 12))

            def waiting() -> bool:
                return not self.stop.is_set() and time.time() < deadline and self._face_present(3.0)

            self.recalibrate()
            while waiting() and self.fer_stream.calibrating:
                time.sleep(0.1)
            with self._stream_lock:
                done = not self.fer_stream.calibrating
                if not done:
                    self.fer_stream.cancel_calibration()
            t_ready = time.time()
            log.info("Cold start: baseline %s", self.fer_stream.calibrator.info())
            n = int(ccfg.get("frames", 12))
            frames: list = []
            while done and waiting() and len(frames) < n:
                frames = self.aggregator.frames_between(t_ready, time.time())
                time.sleep(0.05)
            frames = frames[:n]
            if frames:
                q = [max(1e-3, f.quality) for f in frames]
                v = sum(f.valence * w for f, w in zip(frames, q)) / sum(q)
            if not self._face_present(3.0) and self.utt_q.empty():
                log.info("Cold start: the person left before the greeting")
                self._end_session()
                return
        if self.stop.is_set():
            return
        if self.state == "calibrating":
            self.state = "idle"
        if not self.utt_q.empty():            # already talking to us: answer instead of greeting over it
            return
        key = ("positive" if v >= float(ccfg.get("positive_valence", 0.2)) else
               "negative" if v <= float(ccfg.get("negative_valence", -0.2)) else "neutral")
        log.info("Cold start: mean V %+.2f -> %s greeting", v, key)
        g = GREETINGS[key]
        # the greeting is part of the conversation: "có, hôm nay mình buồn" answers it
        self.memory.add("(người dùng vừa đến trước robot)", f"[EMO:{g.emotion}|{g.intensity}] {g.text}")
        self.say_q.put(g)
        self.say_q.put(END_OF_TURN)

    def _handle(self, utt: Utterance) -> None:
        t_start = time.time()
        self.last_activity = t_start
        affect = fake_affect(self.fake_emotion) if self.fake_emotion else \
            self.aggregator.summarize(utt.t_start, utt.t_end, f"session_{self.memory.session}")
        if affect.get("face_visible"):
            affect["text_vs_face"] = congruence(affect.get("valence"), text_valence(utt.text),
                                                float(self.cfg.get("affect.incongruence_threshold", 0.6)))
        risk = assess_risk(utt.text, affect, self.cfg.section("safety"))
        affect["risk"] = risk
        self.face.set_caption(utt.text)
        log.info("USER: %s | affect=%s", utt.text, affect)
        hist_user = build_history_message(utt.text, affect)

        if risk == "high":   # Immediate Support: local, no LLM round-trip
            seg = Segment("concerned", 0.8, self.cfg.get("safety.support_message"))
            # keep the LLM in the loop next turn: history + Response Guidance (Fluid-Xpress Sec. 3.3)
            self.memory.add(hist_user, f"[EMO:{seg.emotion}|{seg.intensity}] {seg.text}")
            self.memory.guidance = guidance_after_intervention(seg.text)
            self.say_q.put(seg)
            self.say_q.put(END_OF_TURN)
            self._log_turn(utt, affect, seg.text, t_start, None, intervention="immediate_support")
            return

        # what the micro-loop is showing right now, so the first [EMO] tag continues smoothly from it
        robot_face = self.last_micro if self.last_micro and time.time() - self.last_micro_t < 3.0 else None
        guidance = self.memory.take_guidance()
        summary, recent = self.memory.context()
        user_msg = build_user_message(utt.text, affect, robot_face, guidance, summary)
        messages = [{"role": "system", "content": build_system_prompt(self.allowed)},
                    *recent, {"role": "user", "content": user_msg}]
        log.debug("LLM input:\n%s", user_msg)

        spoken, t_first, complete = self._stream_reply(messages)
        if self.stop.is_set():
            return
        if not spoken:
            if guidance:
                self.memory.guidance = guidance    # not delivered: keep it for the next attempt
            self.say_q.put(Segment("concerned", 0.5, APOLOGY))
        else:
            # remember what was actually said, never a half sentence
            self.memory.add(hist_user, " ".join(f"[EMO:{s.emotion}|{s.intensity}] {s.text}" for s in spoken))
            if is_farewell(utt.text):
                self.end_requested = self.wait_absence = True
        self.say_q.put(END_OF_TURN)
        reply = " ".join(f"[EMO:{s.emotion}|{s.intensity}] {s.text}" for s in spoken)
        self._log_turn(utt, affect, reply, t_start, t_first, robot_face=robot_face, guidance=guidance,
                       complete=complete)

    def _stream_reply(self, messages: list[dict]) -> tuple[list[Segment], float | None, bool]:
        """Stream sentences to the speech queue. A stream cut off before any full sentence is retried once;
        a half sentence is never spoken. Returns (spoken segments, time of the first one, complete?)."""
        spoken: list[Segment] = []
        t_first = None
        for attempt in range(2):
            parser = ExpressionStreamParser(self.allowed, self.cfg.get("expression.default", "neutral"))
            info: dict = {}
            try:
                for delta in self.llm.stream(messages, info):
                    if self.stop.is_set():
                        return spoken, t_first, False
                    for seg in parser.feed(delta):
                        t_first = t_first or time.time()
                        spoken.append(seg)
                        self.say_q.put(seg)
            except Exception as e:  # noqa: BLE001 - broken stream or every provider down
                log.warning("LLM stream broke: %s", e)
                info["finish"] = None
            if info.get("finish") == "stop":
                for seg in parser.flush():
                    t_first = t_first or time.time()
                    spoken.append(seg)
                    self.say_q.put(seg)
                if spoken:
                    return spoken, t_first, True
            log.warning("LLM reply incomplete (finish=%s, %d full sentences, attempt %d)",
                        info.get("finish"), len(spoken), attempt + 1)
            if spoken:
                break
        return spoken, t_first, False

    def _speech_loop(self) -> None:
        while not self.stop.is_set():
            try:
                seg = self.say_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.last_activity = time.time()
            if seg is END_OF_TURN:
                self.state = "idle"
                if self.memory.due():   # now nobody is waiting for the LLM
                    threading.Thread(target=self._summarize, name="summary", daemon=True).start()
                continue
            self.state = "speaking"
            self.face.set_expression(seg.emotion, seg.intensity)
            print(f"ROBOT [{seg.emotion}|{seg.intensity:.1f}]: {seg.text}", flush=True)
            try:
                if self.speaker:
                    self.speaker.say(seg.text, on_level=self.face.set_level)
                    continue
            except Exception as e:  # noqa: BLE001 - a sound problem must not silence this thread for good
                self._audio_error(e)
            time.sleep(min(4.0, 0.06 * len(seg.text)))   # simulate speaking time

    def _audio_error(self, e: Exception) -> None:
        if str(e) != self._last_audio_error:          # the full explanation once, not at every sentence
            self._last_audio_error = str(e)
            log.error("Speech output failed, continuing without sound: %s", e)

    # ------------------------------------------------------------------ research log
    def _log_turn(self, utt: Utterance, affect: dict, reply: str, t_start: float, t_first: float | None,
                  robot_face: dict | None = None, guidance: str | None = None, intervention: str | None = None,
                  complete: bool = True):
        rec = {
            "t": utt.t_end, "session": self.memory.session, "text": utt.text, "affect": affect, "reply": reply,
            "micro_loop": self.last_micro, "robot_face_sent": robot_face, "guidance": guidance,
            "intervention": intervention, "complete": complete,
            # sizes only: the summary itself is never written to disk
            "memory": {"turns": len(self.memory.turns), "summary_chars": len(self.memory.summary)},
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
