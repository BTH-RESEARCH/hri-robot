import threading
import time
from typing import ClassVar

import pytest

from hri.audio.types import Utterance
from hri.config import load_config
from hri.dialogue.memory import SessionMemory, is_farewell


# ---------------------------------------------------------------- memory
def test_bounded_summary_keeps_recent_turns():
    m = SessionMemory({"keep_turns": 6, "summarize_every": 4})
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return f"[EMO:calm|0.5] tóm tắt {len(seen)}"

    for i in range(50):
        m.add(f"u{i}", f"[EMO:happy|0.5] a{i}")
        if m.due():
            assert m.compact(complete)
        assert len(m.turns) < 10
    assert m.turns[-1][0] == "u49" and len(m.turns) >= 6
    assert m.summary == f"tóm tắt {len(seen)}"         # tags removed
    assert "tóm tắt 1" in seen[1] and "[EMO" not in seen[0].split("Các lượt mới:")[1]
    # every turn is either summarised once or still verbatim
    assert [t[0] for t in m.turns] == [f"u{i}" for i in range(len(seen) * 4, 50)]


def test_failed_or_runaway_summary_stays_bounded():
    m = SessionMemory({"max_turns": 14})

    def boom(_):
        raise RuntimeError("503")

    for i in range(30):
        m.add(f"u{i}", "a")
        if m.due():
            assert not m.compact(boom)
    assert len(m.turns) == 14 and m.turns[-1][0] == "u29" and m.summary == ""
    assert m.compact(lambda _: "từ " * 1000)
    assert len(m.summary.split()) == 160


def test_late_summary_never_leaks_into_next_session():
    m = SessionMemory()
    for i in range(10):
        m.add(f"u{i}", "a")
    go, res = threading.Event(), []
    t = threading.Thread(target=lambda: res.append(m.compact(lambda _: (go.wait(5), "bí mật người trước")[1])))
    t.start()
    m.reset()
    m.add("người mới", "chào")
    go.set()
    t.join(5)
    assert res == [False] and m.summary == "" and m.turns == [("người mới", "chào")]


@pytest.mark.parametrize("text, expected", [
    ("tạm biệt nhé", True), ("cảm ơn bạn, tạm biệt grace", True), ("thôi mình về đây", True), ("bye bye", True),
    ("mình vừa tạm biệt bạn thân", False), ("mình đi học về rồi", False),
    ("tạm biệt người yêu cũ làm mình buồn suốt cả tuần", False), ("phần đầu", False),
])
def test_farewell(text, expected):
    assert is_farewell(text) is expected


# ---------------------------------------------------------------- app lifecycle with a scripted LLM
class FakeLLM:
    script: ClassVar[list] = []

    def __init__(self, cfg):
        self.calls = []

    def stream(self, messages, info=None):
        self.calls.append(messages)
        text, finish = FakeLLM.script.pop(0) if FakeLLM.script else ("[EMO:happy|0.5] Vâng.", "stop")
        yield text
        info["finish"] = finish

    def complete(self, messages):
        return "tóm tắt"


class FakeVision:
    pass


@pytest.fixture
def app(monkeypatch, tmp_path):
    import hri.llm.client
    from hri.dialogue.app import RobotApp
    FakeLLM.script = []
    monkeypatch.setattr(hri.llm.client, "LLMClient", FakeLLM)
    cfg = load_config("laptop", overrides=[f"paths.data_dir={tmp_path.as_posix()}"])
    a = RobotApp(cfg, text_mode=True, use_vision=False, use_tts=False, use_ui=False)
    yield a
    a.stop.set()


def say(a, text):
    now = time.time()
    a._handle(Utterance(text, now - 1, now))
    a._drain(a.say_q)


def test_follow_up_sees_previous_turns(app):
    app._start_session()
    app._greet()
    say(app, "kể chuyện tấm cám")
    say(app, "phần đầu")
    sent = app.llm.calls[-1]
    text = " ".join(m["content"] for m in sent[1:])
    assert "Grace" in text and "kể chuyện tấm cám" in text and "phần đầu" in sent[-1]["content"]


def test_truncated_reply_is_retried_and_never_spoken(app):
    app._start_session()
    FakeLLM.script = [("[EMO:cal", None), ("[EMO:calm|0.6] Mình nghe đây.", "stop")]
    now = time.time()
    app._handle(Utterance("ổn", now - 1, now))
    segs = [s for s in list(app.say_q.queue) if s is not None]
    assert [s.text for s in segs] == ["Mình nghe đây."]
    app._drain(app.say_q)
    # cut off after one full sentence: keep the sentence, drop the half one
    FakeLLM.script = [("[EMO:calm|0.6] Câu đủ. Mình thấy nét", None)]
    say(app, "rồi sao")
    assert app.memory.turns[-1][1] == "[EMO:calm|0.6] Câu đủ."


def test_session_ends_only_without_face_and_speech(app):
    app.vision = FakeVision()
    app._start_session()
    say(app, "mình tên Lan")
    now = time.time()
    app.last_activity, app.last_face = now - 100, now
    assert not app._session_over()                      # silent but still standing there
    app.last_activity, app.last_face = now, now - 100
    assert not app._session_over()                      # face lost but talking
    app.last_activity = now - 100
    app.state = "thinking"                              # VAD fired on noise, ASR found no text
    assert app._session_over()
    app._end_session()
    assert not app.active and app.memory.turns == [] and app.state == "idle"
    app._wait_for_user()
    assert not app.active                               # nobody there
    app.utt_q.put(Utterance("xin chào", now, now))
    app._wait_for_user()
    assert app.active
    say(app, "xin chào")
    assert "Lan" not in str(app.llm.calls[-1][1:])       # nothing leaks to the next person


def test_farewell_ends_after_speaking_and_does_not_regreet(app):
    app.vision = FakeVision()
    app._start_session()
    app.last_face = time.time()
    now = time.time()
    app._handle(Utterance("tạm biệt nhé", now - 1, now))
    assert not app._session_over()                      # goodbye still queued
    app._drain(app.say_q)
    assert app._session_over()
    app._end_session()
    app._wait_for_user()
    assert not app.active                               # same face still in view: no new greeting
    app.last_face = time.time() - 10
    app._wait_for_user()
    assert not app.wait_absence


def test_greeting_gives_up_when_person_leaves(app):
    app.vision = FakeVision()
    app._start_session()
    app.last_face = time.time() - 10
    app._greet()
    assert not app.active and app.say_q.empty() and not app.fer_stream.calibrating


def test_guidance_survives_failed_turn(app):
    app._start_session()
    say(app, "mình muốn chết")
    FakeLLM.script = [("[EMO:c", None), ("[EMO:c", None)]
    say(app, "mình sợ")
    assert app.memory.guidance
    say(app, "bạn còn đó không")
    assert "GUIDANCE" in app.llm.calls[-1][-1]["content"] and app.memory.guidance is None
