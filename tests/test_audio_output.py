import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hri.audio.output import Output, Player, convert
from hri.config import load_config
from hri.llm.expression_parser import Segment


def fake_sd(accept):
    calls = []

    def stream(**kw):
        calls.append((kw["device"], kw["samplerate"], kw["channels"]))
        if not accept(*calls[-1]):
            raise OSError("Device unavailable [PaErrorCode -9985]")
        return SimpleNamespace(**kw)
    return SimpleNamespace(OutputStream=stream, calls=calls)


def test_falls_back_to_hdmi_stereo_and_remembers():
    sd = fake_sd(lambda d, r, c: d == "sysdefault" and r == 48000 and c == 2)
    out = Output(None, sd=sd, platform="linux")
    assert out.open(22050)[1:] == (48000, 2)
    n = len(sd.calls)
    out.open(22050)
    assert len(sd.calls) == n + 1                    # straight to the working combination


def test_error_lists_reasons_and_help():
    with pytest.raises(RuntimeError, match="fuser"):
        Output("x", sd=fake_sd(lambda *a: False), platform="linux").open(22050)


def test_convert_keeps_pitch_and_duration():
    sr = 22050
    x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    y = convert(x, sr, 48000, 2)
    assert y.shape == (48000, 2) and y.dtype == np.float32 and np.array_equal(y[:, 0], y[:, 1])
    assert np.argmax(np.abs(np.fft.rfft(y[:, 0]))) == 440
    assert convert(x, sr, sr, 1).shape == (sr, 1)


def test_ui_never_opens_the_sound_card():
    code = ("import pygame as pg\nfrom hri.ui.face_renderer import FaceRenderer\n"
            "from hri.ui.expressions import FaceController\n"
            "FaceRenderer({'width': 320, 'height': 200}, FaceController()).step(0.1)\n"
            "assert pg.mixer.get_init() is None\n")
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_speech_thread_survives_audio_failure():
    from hri.dialogue.app import END_OF_TURN, RobotApp
    from hri.ui.expressions import FaceController
    a = RobotApp.__new__(RobotApp)
    import queue
    a.cfg, a._last_distress, a.fake_emotion = load_config("laptop"), -1e9, "neutral"
    a.stop, a.say_q, a.face, a._last_audio_error = threading.Event(), queue.Queue(), FaceController(), ""
    a.memory, a.state, tried = SimpleNamespace(due=lambda: False), "idle", []

    def broken(text, on_level=None, pace=1.0):
        tried.append(text)
        raise RuntimeError("Device unavailable")
    a.speaker, a.asr = SimpleNamespace(say=broken, end_turn=lambda: None), None
    t = threading.Thread(target=a._speech_loop, daemon=True)
    t.start()
    for x in ("a", "b"):
        a.say_q.put(Segment("happy", 0.5, x))
    a.say_q.put(END_OF_TURN)
    deadline = time.time() + 5
    while (len(tried) < 2 or a.state != "idle") and time.time() < deadline:
        time.sleep(0.02)
    assert tried == ["a", "b"] and a.state == "idle" and t.is_alive()
    a.stop.set()


# ---------------------------------------------------------------- nothing of a sentence may be lost
class FakeStream:
    """Calls the callback from a thread like PortAudio; keeps everything that was 'played'."""
    opened = 0

    def __init__(self, **kw):
        FakeStream.opened += 1
        self.kw, self.out, self.latency, self._run = kw, [], 0.05, threading.Event()

    def start(self):
        def loop():
            while not self._run.is_set():
                o = np.ones((256, self.kw["channels"]), np.float32)
                self.kw["callback"](o, 256, None, None)
                self.out.append(o[:, 0].copy())
                time.sleep(0.001)
        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()

    def stop(self):
        self._run.set()
        self.t.join()

    def close(self):
        FakeStream.last = np.concatenate(self.out)


def player(**kw):
    FakeStream.opened = 0
    return Player(Output(None, sd=SimpleNamespace(OutputStream=lambda **k: FakeStream(**k)), platform="win32"), **kw)


def test_turn_plays_every_sample_with_one_stream_and_a_silent_tail():
    p = player(tail_sec=0.1)
    a, b = np.full(5000, 0.5, np.float32), np.full(3001, 0.25, np.float32)
    levels = []
    assert p.play(a, 22050, levels.append) and p.play(b, 22050, levels.append)
    assert p.is_open and FakeStream.opened == 1
    p.close()
    got = FakeStream.last
    assert (got == 0.5).sum() == 5000 and (got == 0.25).sum() == 3001      # up to the very last sample
    end = np.where(got != 0)[0][-1]
    assert len(got) - end > 2000 and max(levels) > 0.5 and levels[-1] == 0.0
    p.play(a, 22050)
    assert FakeStream.opened == 2                                          # next turn
    p.close()


def test_interrupt_stops_quickly_and_player_still_works():
    p = player(tail_sec=0.0)
    threading.Timer(0.05, p.interrupt).start()
    t0 = time.time()
    assert p.play(np.full(22050 * 60, 0.5, np.float32), 22050) is False and time.time() - t0 < 1
    assert p.play(np.full(500, 0.5, np.float32), 22050) is True
    p.close()


# ---------------------------------------------------------------- first word
class FakeVad:
    """Detects |x| > 0.3 and, like silero on soft consonants, reports the onset too late."""
    LATE = 3200

    def __init__(self):
        self.reset()

    def reset(self):
        self.n, self.cur, self.segs = 0, None, []

    def accept_waveform(self, w):
        loud = np.abs(w).max() > 0.3
        if loud and self.cur is None:
            self.cur = [self.n, []]
        if self.cur is not None:
            self.cur[1].append(w)
            if not loud:
                body = np.concatenate(self.cur[1])
                self.segs.append(SimpleNamespace(start=self.cur[0] + self.LATE, samples=body[self.LATE:]))
                self.cur = None
        self.n += len(w)

    def is_speech_detected(self):
        return self.cur is not None

    def empty(self):
        return not self.segs

    front = property(lambda self: self.segs[0])

    def pop(self):
        self.segs.pop(0)


def run_seg(seg, x, muted=False):
    out = []
    for i in range(0, len(x), 300):
        out += [s for e, s in seg.feed(x[i:i + 300], muted) if e == "end"]
    return out


def test_pre_roll_restores_the_late_onset_but_never_the_previous_utterance():
    from hri.audio.asr import Segmenter
    seg = Segmenter(FakeVad(), 16000, pre_roll=0.4, guard=0.0)
    q, sp = np.full(16000, 0.01, np.float32), np.full(16000, 0.5, np.float32)
    first, second = run_seg(seg, np.concatenate([q, sp, q, np.full(16000, 0.6, np.float32), q]))
    assert (first == 0.5).sum() >= 15900 and (second == np.float32(0.6)).sum() >= 15900
    assert not (second == 0.5).any()
    # long silence: positions stay right although old audio is forgotten
    late = run_seg(seg, np.concatenate([np.full(16000 * 40, 0.01, np.float32), sp, q]))
    assert (late[0] == 0.5).sum() >= 15900


def test_nothing_said_by_the_robot_reaches_the_recogniser():
    from hri.audio.asr import Segmenter
    seg = Segmenter(FakeVad(), 16000, pre_roll=0.4, guard=0.15)
    q, robot, parent = (np.full(16000, v, np.float32) for v in (0.01, 0.9, 0.5))
    assert run_seg(seg, robot, muted=True) == []
    got = run_seg(seg, np.concatenate([q[:4000], parent, q]))
    assert len(got) == 1 and not (got[0] == np.float32(0.9)).any() and (got[0] == 0.5).sum() >= 15900


def test_real_vad_positions_match_segmenter():
    """Segmenter relies on silero positions counting from reset()."""
    from hri.audio.asr import Segmenter, build_vad
    from hri.models import ModelRegistry
    cfg = load_config("laptop")
    reg = ModelRegistry(cfg)
    if not reg.is_present("silero_vad") or not reg.is_present("asr_vi"):
        pytest.skip("models not downloaded")
    import wave
    f = sorted((reg.path("asr_vi") / "test_wavs").glob("*.wav"))[1]
    with wave.open(str(f)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    x = np.concatenate([np.zeros(16000, np.float32), x, np.zeros(16000, np.float32)])
    for pre in (0.0, 0.4):
        seg = Segmenter(build_vad(reg, cfg.section("audio.vad"), 16000), 16000, pre, 0.0)
        for _ in range(2):      # also right after a robot turn
            got = run_seg(seg, x)
            assert got and any(np.array_equal(got[0], x[k:k + len(got[0])]) for k in range(0, 16000 * 3, 1))
            seg.feed(np.zeros(512, np.float32), True)


def test_microphone_is_closed_for_the_whole_turn():
    from hri.dialogue.app import RobotApp
    a = RobotApp.__new__(RobotApp)
    a.speaker = SimpleNamespace(playing=threading.Event())
    a.state = "speaking"                                   # between two sentences nothing is playing
    assert a._asr_muted()
    a.state = "idle"
    assert not a._asr_muted()
    a.speaker.playing.set()                                # device still playing its tail
    assert a._asr_muted()
