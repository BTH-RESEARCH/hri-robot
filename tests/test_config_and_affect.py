from hri.affect.child_state import GROUPS
from hri.config import apply_overrides, deep_merge, load_config
from hri.dialogue.app import fake_child
from hri.llm.prompts import build_history_message, build_system_prompt, build_user_message


def test_profiles_load_and_merge():
    lap = load_config("laptop")
    pi = load_config("pi5", overrides=["llm.temperature=0.2", "ui.fullscreen=false"])
    assert lap.get("camera.backend") == "opencv"
    assert pi.get("camera.backend") == "picamera2"
    assert pi.get("vision.fer.model") == "fer"
    assert pi.get("vision.child_zone.width") == 0.5
    assert set(pi.get("micro_loop.pace")) == set(GROUPS)
    assert pi.get("llm.temperature") == 0.2 and pi.get("ui.fullscreen") is False


def test_deep_merge_and_overrides():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert apply_overrides({}, ["x.y=[1, 2]"]) == {"x": {"y": [1, 2]}}


def test_system_prompt():
    s = build_system_prompt(["happy", "sad"])
    for k in ("happy, sad", "4 đến 5 câu", "2 đến 3 câu", "[TOPIC:", "[EMO:", "CHILD", "REACTION", "MEMORY", "GUIDANCE", *GROUPS):
        assert k in s
    assert "{" in s and "{{" not in s


def test_user_message_and_history():
    c = fake_child("happiness")
    lines = build_user_message("kể tiếp", c, {"topic": "con mèo", "verdict": "liked", "summary": {}}, ["con mèo"], [],
                               {"expression": "happy"}, "G", "M").split("\n")
    assert [x.split(":")[0] for x in lines] == ["MEMORY", "CHILD", "REACTION", "LIKES", "ROBOT_FACE", "GUIDANCE", "Phụ huynh"]
    assert "shares" not in lines[1] and "summary" not in lines[2]
    assert build_user_message("a", {"visible": False, "reason": "away"}).startswith(
        'CHILD: {"visible": false, "reason": "away"}')
    assert "thích thú" in build_history_message("a", c)
    assert build_history_message("a", {**c, "calibrated": False}) == "Phụ huynh: a"
