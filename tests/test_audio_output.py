import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hri.audio.output import Output, convert
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
    a.stop, a.say_q, a.face, a._last_audio_error = threading.Event(), queue.Queue(), FaceController(), ""
    a.memory, a.state, tried = SimpleNamespace(due=lambda: False), "idle", []

    def broken(text, on_level=None):
        tried.append(text)
        raise RuntimeError("Device unavailable")
    a.speaker = SimpleNamespace(say=broken)
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
