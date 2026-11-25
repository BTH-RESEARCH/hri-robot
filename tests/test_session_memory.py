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
    segs = [s for s in list(app.say_q.queue) if hasattr(s, "text")]
    assert [s.text for s in segs] == ["Mình nghe đây."]
    app._drain(app.say_q)
    # cut off after one full sentence: keep the sentence, drop the half one
    FakeLLM.script = [("[EMO:calm|0.6] Câu đủ. Mình thấy nét", None), *[("", None)] * 3]
    say(app, "rồi sao")
    assert app.memory.turns[-1][1] == "[TOPIC:khác] [EMO:calm|0.6] Câu đủ."
    assert len(app.llm.calls) == 2 + 4                  # bounded


def test_dropped_stream_is_continued_after_what_was_said(app):
    app._start_session()
    FakeLLM.script = [("[TOPIC:con mèo] [EMO:happy|0.6] Câu một. [EMO:calm|0.6] Chú mè", None),
                      ("[TOPIC:khác hẳn] [EMO:calm|0.6] Câu hai. [EMO:ca", None), ("[EMO:calm|0.5] Câu ba.", "stop")]
    say(app, "kể về con mèo")
    assert app.memory.turns[-1][1] == ("[TOPIC:con mèo] [EMO:happy|0.6] Câu một. [EMO:calm|0.6] Câu hai. "
                                       "[EMO:calm|0.5] Câu ba.")
    assert app.llm.calls[-1][-2] == {"role": "assistant", "content":
                                     "[TOPIC:con mèo] [EMO:happy|0.6] Câu một. [EMO:calm|0.6] Câu hai."}


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


def test_guidance_survives_failed_turn(app):
    app._start_session()
    say(app, "mình muốn chết")
    FakeLLM.script = [("[EMO:c", None)] * 4
    say(app, "mình sợ")
    assert app.memory.guidance
    say(app, "bạn còn đó không")
    assert "GUIDANCE" in app.llm.calls[-1][-1]["content"] and app.memory.guidance is None


# ---------------------------------------------------------------- child in the loop
def test_reaction_reaches_next_prompt(app):
    app.fake_emotion = "happiness"
    app._start_session()
    FakeLLM.script = [("[TOPIC:con mèo] [EMO:happy|0.5] Meo meo.", "stop")]
    now = time.time()
    app._handle(Utterance("kể về con mèo", now - 1, now))
    assert list(app.say_q.queue)[-1].topic == "con mèo"
    app._observe("con mèo", now, now + 1, app.memory.session)
    app._drain(app.say_q)
    say(app, "kể tiếp")
    msg = app.llm.calls[-1][-1]["content"]
    assert "LIKES: con mèo" in msg and "enjoy_curious" in msg
    app._end_session()
    app._start_session()
    say(app, "chào")
    assert "con mèo" not in str(app.llm.calls[-1][1:])


def test_talking_to_parent_is_not_content(app):
    app._start_session()
    FakeLLM.script = [("[TOPIC:phụ huynh] [EMO:calm|0.5] Dạ.", "stop")]
    now = time.time()
    app._handle(Utterance("dạy con thế nào", now - 1, now))
    assert list(app.say_q.queue)[-1].topic is None


def test_soothe_interrupts_reply_and_tells_parent(app):
    app._start_session()

    class Slow(FakeLLM):
        def stream(self, messages, info=None):
            yield "[TOPIC:sư tử] [EMO:excited|0.8] Gầm! "
            app._soothe()
            yield "[EMO:excited|0.8] Gầm nữa! "
            info["finish"] = "stop"
    app.llm = Slow(None)
    now = time.time()
    app._handle(Utterance("kể sư tử", now - 1, now))
    texts = [getattr(s, "text", "<end>") for s in list(app.say_q.queue)]
    assert texts == [app.cfg.get("child.distress.soothe"), app.cfg.get("child.distress.tell_parent"), "<end>"]
    app._drain(app.say_q)
    app.llm = FakeLLM(None)
    say(app, "kể cái khác")                              # the next command works and knows what happened
    assert "GUIDANCE" in app.llm.calls[-1][-1]["content"]


def test_log_has_no_memory_text(app):
    app._start_session()
    say(app, "kể chuyện")
    rec = __import__("json").loads(app.turn_log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["event"] == "turn" and "summary_chars" in rec["memory"] and "child" in rec


def test_distress_counts_against_topic_only_while_speaking(app):
    app._start_session()
    app._topic, app.state = "sư tử", "idle"
    app._soothe()
    assert app.content.items == []
    app.state = "speaking"
    app._soothe()
    assert [i["topic"] for i in app.content.items] == ["sư tử"]


# ---------------------------------------------------------------- vague commands
@pytest.mark.parametrize("text, topic, vague", [
    ("chào đi", "", True), ("Nói gì đi.", "", True), ("kể chuyện", "", True), ("kể chuyện cho bé nghe đi Grace", "khác", True),
    ("chào bé đi Grace", "", True), ("kể tiếp đi", "", True), ("kể tiếp đi", "con mèo", False), ("kể chuyện con mèo", "", False),
    ("ừ kể đi", "", False), ("bé nghe gì", "", False), ("tạm biệt nhé", "", False), ("bé hay khóc đêm", "", False),
])
def test_vague_command(text, topic, vague):
    from hri.llm.prompts import guidance_vague
    assert (guidance_vague(text, topic) is not None) is vague


def test_vague_hint_once_per_session(app):
    app._start_session()
    FakeLLM.script = [("[TOPIC:chào hỏi] [EMO:happy|0.5] Chào bé.", "stop")]
    say(app, "chào đi")
    assert "GUIDANCE: Phụ huynh chưa nêu chủ đề" in app.llm.calls[-1][-1]["content"]
    say(app, "kể chuyện đi")                       # still vague: now the LLM chooses, no endless asking
    assert "GUIDANCE" not in app.llm.calls[-1][-1]["content"]
    app._end_session()
    app._start_session()
    say(app, "kể chuyện đi")
    assert "GUIDANCE" in app.llm.calls[-1][-1]["content"]


def test_system_prompt_formats():
    from hri.llm.prompts import build_system_prompt
    p = build_system_prompt(["happy", "calm"])
    assert "happy, calm" in p and '{"visible": true' in p and "{{" not in p and "miệng" not in p.split("# Vai trò")[0]
