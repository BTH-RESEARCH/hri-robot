import numpy as np
import pytest

from hri.affect.aggregator import AffectAggregator, congruence, emotional_stability
from hri.affect.risk import assess_risk
from hri.affect.text_sentiment import text_valence
from hri.config import apply_overrides, deep_merge, load_config
from hri.llm.prompts import build_history_message, build_system_prompt, build_user_message

LABELS = ["neutral", "happy", "sad"]


def test_profiles_load_and_merge():
    lap = load_config("laptop")
    pi = load_config("pi5", overrides=["llm.temperature=0.2", "ui.fullscreen=false"])
    assert lap.get("camera.backend") == "opencv"
    assert pi.get("camera.backend") == "picamera2"
    assert pi.get("vision.fer.model") == "fer"           # fp32 on both devices
    assert pi.get("vision.baseline.calib_seconds") == 3.0  # inherited from default
    assert "fer" in pi.get("models.required")
    assert pi.get("llm.temperature") == 0.2 and pi.get("ui.fullscreen") is False


def test_deep_merge_and_overrides():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert apply_overrides({}, ["x.y=[1, 2]"]) == {"x": {"y": [1, 2]}}


def _msg(t, probs, val):
    return {"t": t, "probs": probs, "valence": val, "arousal": 0.1, "confidence": max(probs), "quality": 1.0}


def test_aggregator_summary_and_speaking_weight():
    agg = AffectAggregator({"speaking_weight": 0.1, "min_frames": 3, "pre_utterance_sec": 0}, LABELS)
    for i in range(10):      # listening: sad
        agg.add(_msg(100 + i * 0.1, [0.1, 0.1, 0.8], -0.6))
    for i in range(10):      # speaking: looks happy (mouth motion) -> down-weighted
        agg.add(_msg(101 + i * 0.1, [0.1, 0.8, 0.1], 0.6))
    agg.mark_speaking(101, 102)
    s = agg.summarize(100, 102, speaker="lan")
    assert s["face_visible"] and s["dominant"] == "sad" and s["valence"] < 0
    assert s["vs_baseline"] == "unknown"
    assert agg.summarize(200, 201) == {"face_visible": False}


def test_congruence_text_sentiment_and_risk():
    assert text_valence("hôm nay mình rất vui") > 0
    assert text_valence("mình không vui") < 0
    assert congruence(-0.5, 0.8, 0.6) == "incongruent"
    assert congruence(0.5, 0.8, 0.6) == "congruent"
    safety = {"enabled": True, "keywords": ["muốn chết"], "negative_valence_threshold": -0.6}
    assert assess_risk("mình muốn chết", {}, safety) == "high"
    assert assess_risk("ok", {"face_visible": True, "valence": -0.8, "stability": 0.9}, safety) == "elevated"
    assert assess_risk("ok", {"face_visible": True, "valence": -0.8, "stability": None}, safety) == "low"


# ---------------------------------------------------------------- Emotional Stability (paper Eq. 1-2)
def test_es_steady_vs_volatile_and_fps_independent():
    cfg = {"m_max": 0.1, "p_max": 0.3, "acute_threshold": 0.15, "resample_hz": 5, "max_gap_sec": 0.5}
    for fps in (8, 15):                                   # Pi5 vs laptop
        ts = np.arange(0, 4, 1 / fps)
        steady = emotional_stability(ts, np.full(len(ts), -0.3), np.full(len(ts), 0.1), cfg)
        zigzag = emotional_stability(ts, 0.4 * np.sin(2 * np.pi * 1.0 * ts), np.zeros(len(ts)), cfg)
        assert steady["es"] == pytest.approx(1.0) and steady["mssd"] == 0
        assert zigzag["es"] < 0.5
    # resampling makes MSSD nearly identical across frame rates
    es8 = emotional_stability(np.arange(0, 4, 1 / 8), 0.4 * np.sin(2 * np.pi * np.arange(0, 4, 1 / 8)),
                              np.zeros(32), cfg)
    ts15 = np.arange(0, 4, 1 / 15)
    es15 = emotional_stability(ts15, 0.4 * np.sin(2 * np.pi * ts15), np.zeros(len(ts15)), cfg)
    assert es8["mssd"] == pytest.approx(es15["mssd"], rel=0.25)


def test_es_ignores_face_loss_gaps():
    cfg = {"resample_hz": 0, "max_gap_sec": 0.5, "acute_threshold": 0.15}
    ts = np.array([0, 0.1, 0.2, 5.0, 5.1, 5.2])            # face lost for ~5 s
    v = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])          # different level after the gap
    assert emotional_stability(ts, v, np.zeros(6), cfg)["pac"] == 0.0


# ---------------------------------------------------------------- prompt / turn message
def test_system_prompt_contains_labels():
    s = build_system_prompt(["happy", "sad"], user_name="Lan")
    assert "happy, sad" in s and "Lan" in s and "[EMO:" in s
    assert "sadness" in s and '"dominant": "sad"' not in s       # examples use FERPlus names
    for field in ("arousal", "stability", "ROBOT_FACE", "GUIDANCE", "vs_baseline"):
        assert field in s
    assert "user" not in build_system_prompt(["happy"], user_name="user").split("\n")[-2]


def test_user_message_and_history():
    affect = {"face_visible": True, "dominant": "sadness", "valence": -0.4, "confidence": 0.7,
              "stability": None, "n_frames": 20, "mssd": 0.01, "pac": 0.1}
    msg = build_user_message("Mình ổn", affect, {"expression": "calm", "strategy": "de-escalation"}, "G")
    lines = msg.split("\n")
    assert lines[0].startswith("AFFECT:") and "n_frames" not in msg and "mssd" not in msg
    assert "stability" not in lines[0]                           # None values dropped
    assert lines[1] == "ROBOT_FACE: calm (de-escalation)" and lines[2] == "GUIDANCE: G"
    assert lines[3] == "Người dùng: Mình ổn"
    assert build_user_message("hi", {"face_visible": False, "n_frames": 0}) == \
        'AFFECT: {"face_visible": false}\nNgười dùng: hi'
    assert build_history_message("Mình ổn", affect) == "(nét mặt: sadness) Người dùng: Mình ổn"
    assert build_history_message("hi", {"face_visible": False}) == "Người dùng: hi"
