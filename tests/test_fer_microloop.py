import zipfile
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hri.dialogue.micro_loop import ATTENTIVE, DEESCALATION, MIRRORING, SUPPORT, decide
from hri.models.registry import _safe_extract
from hri.vision.fer import FERPLUS_CLASSES, preprocess_rgb, square_crop, va_from_probs, va_table
from hri.vision.postproc import (
    BaselineCalibrator,
    EMASmoother,
    FERStream,
    HysteresisSegmenter,
    SensitivityScaler,
)

L = FERPLUS_CLASSES  # neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
META = {"img_size": 224, "grayscale": True, "train_res": 48,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
MICRO = {"min_frames": 3, "positive_valence": 0.2, "negative_valence": -0.2, "high_arousal": 0.5,
         "min_intensity": 0.3, "max_intensity": 0.8}


def onehot(name, p=0.9):
    v = np.full(len(L), (1 - p) / (len(L) - 1), np.float32)
    v[L.index(name)] = p
    return v


# ---------------------------------------------------------------- preprocessing (same as fer_demo)
def test_preprocess_matches_training_transform():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
    x = preprocess_rgb(rgb, META)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    # reference: exactly fer_demo FERClassifier.to_train_like + preprocess
    img = Image.fromarray(rgb).convert("L").resize((48, 48), Image.BOX).convert("RGB").resize((224, 224), Image.BILINEAR)
    ref = (np.asarray(img, np.float32) / 255 - META["mean"]) / META["std"]
    assert np.allclose(x[0], ref.transpose(2, 0, 1), atol=1e-6)
    assert np.allclose(x[0, 0] * 0.229 + 0.485, x[0, 1] * 0.224 + 0.456, atol=1e-5)  # gray -> equal channels


def test_square_crop_like_fer_demo():
    img = np.zeros((480, 640, 3), np.uint8)
    crop = square_crop(img, (300, 200, 100, 80), 0.15)   # YuNet x, y, w, h
    assert crop.shape[:2] == (130, 130)                   # side = max(w, h) * (1 + 2 * margin)
    assert square_crop(img, (600, 450, 100, 100), 0.15).shape[:2] == (480 - 435, 640 - 585)  # clipped


def test_valence_arousal_from_probs():
    tab = va_table(L)
    v, a = va_from_probs(onehot("happiness", 1.0), tab)
    assert (v, a) == pytest.approx((0.8, 0.5))
    v, _ = va_from_probs(onehot("sadness", 1.0), tab)
    assert v < -0.5


# ---------------------------------------------------------------- post-processing (fer_demo v2 / V3b)
def test_sensitivity_scaling():
    sc = SensitivityScaler(L, {"neutral": 0.5, "sadness": 2.0})
    p = np.full(len(L), 1 / len(L), np.float32)
    out = sc(p)
    w = np.array([0.5, 1, 1, 2.0, 1, 1, 1, 1])
    assert np.allclose(out, w / w.sum())
    assert np.allclose(SensitivityScaler(L, enabled=False)(p), p)


def test_ema_optional():
    s = EMASmoother(alpha=0.45)
    a, b = onehot("neutral"), onehot("happiness")
    s(a)
    assert np.allclose(s(b), 0.45 * b + 0.55 * a)


def test_baseline_sanity_clamp_and_dynamic_gain():
    gains = {"neutral": 1.0, "happiness": 1.6, "surprise": 2.0, "sadness": 3.2, "anger": 3.2,
             "fear": 3.4, "contempt": 3.4, "disgust": 3.6}
    cal = BaselineCalibrator(L, calib_seconds=1.0, min_samples=5, max_non_neutral=0.15,
                             neutral_margin=0.20, class_gains=gains)
    rest = onehot("sadness", 0.4)                        # user starts the session already sad
    done = [cal.update_calibration(rest, 0.25) for _ in range(5)]
    assert done[-1] and cal.is_calibrated
    assert cal.p_base[L.index("sadness")] <= 0.15 + 1e-6  # clamped, surplus moved to neutral
    assert cal.p_base.sum() == pytest.approx(1.0)
    p = onehot("sadness", 0.3)
    p[0] = 0.5
    p /= p.sum()
    out = cal.adjust(p)
    # demo formula (calibration.py): gains per class, neutral - margin - 2.2 * max positive delta
    delta = p - cal.p_base
    g = np.array([gains[c] for c in L])
    exp = np.where(delta > 0, p + delta * g, p * 0.5)
    m = max(0.0, float(np.delete(delta, 0).max()))
    exp[0] = max(0.0, p[0] - 0.20 - (2.2 * m if m > 0.015 else 0.0))
    assert np.allclose(out, exp / exp.sum(), atol=1e-6)
    assert L[int(out.argmax())] == "sadness"           # sadness above the clamped baseline surfaces


def _run(seg, frames, t0=0.0, dt=1 / 15):
    labels = []
    for i, p in enumerate(frames):
        labels.append(seg.update(t0 + i * dt, p))
    return labels


def test_hysteresis_switches_after_t_min_with_class_threshold():
    seg = HysteresisSegmenter(L)
    seg.update(0.0, onehot("neutral", 0.9))
    sad = onehot("sadness", 0.30)                       # only 0.30 > sadness start threshold 0.24
    labels = _run(seg, [sad] * 10, t0=0.1)
    assert labels[0] == "neutral" and labels[-1] == "sadness"
    assert labels.index("sadness") >= 3                 # needs t_min = 0.25 s (~4 frames at 15 FPS)


def test_hysteresis_blocks_opposite_jump_and_needs_neutral_bridge():
    seg = HysteresisSegmenter(L)
    _run(seg, [onehot("sadness", 0.8)] * 10)
    assert seg.active_state == "sadness"
    # sadness -> happiness: barrier 0.35 => needs >= 0.38 + 0.35 = 0.73; 0.6 is not enough
    happy = onehot("happiness", 0.6)
    happy[L.index("sadness")] = 0.01
    happy /= happy.sum()
    dt = 1 / 15
    labels = _run(seg, [happy] * 30, t0=1.0, dt=dt)
    assert labels[-1] == "happiness"
    # never a direct sadness -> happiness switch: it must pass through neutral ...
    pairs = {(a, b) for a, b in pairwise(labels) if a != b}
    assert pairs == {("sadness", "neutral"), ("neutral", "happiness")}
    # ... for at least neutral_bridge_s (0.25 s) before happiness may start
    n_first, h_first = labels.index("neutral"), labels.index("happiness")
    assert (h_first - n_first) * dt >= 0.25


def test_hysteresis_fade_grace_ignores_one_frame_drop():
    seg = HysteresisSegmenter(L)
    _run(seg, [onehot("happiness", 0.8)] * 10)
    drop = onehot("neutral", 0.5)                       # happiness below its end threshold for 1 frame
    assert seg.update(1.0, drop) == "happiness"          # within fade_grace_s = 0.2 s
    assert seg.update(1.05, onehot("happiness", 0.8)) == "happiness"


def test_fer_stream_stable_label_and_va():
    st = FERStream(L, {"target_fps": 15, "baseline": {"enabled": False}})
    t = 0.0
    for _ in range(10):
        f = st.update({"type": "fer", "t": t, "probs": onehot("happiness", 0.9).tolist(), "quality": 1.0})
        t += 1 / 15
    assert f.label == "happiness" and f.top == "happiness" and f.valence > 0.5
    assert f.to_msg(L)["label"] == "happiness"


def test_fer_stream_calibrates_on_face_time_only():
    st = FERStream(L, {"baseline": {"calib_seconds": 1.0, "min_samples": 3}})
    st.start_calibration()
    t = 100.0
    for _ in range(5):
        t += 0.2
        st.update({"type": "fer", "t": t, "probs": onehot("neutral").tolist(), "quality": 0.9})
    assert st.calibrating                         # 0.8 s of face time so far
    st.update({"type": "no_face", "t": t + 5})    # a gap must not count
    for _ in range(6):
        t += 0.2
        f = st.update({"type": "fer", "t": t + 5, "probs": onehot("neutral").tolist(), "quality": 0.9})
    assert not st.calibrating and f.calibrated


# ---------------------------------------------------------------- micro-loop strategies
def frames(name, n=8, p=0.9):
    tab = va_table(L)
    pr = onehot(name, p)
    v, a = va_from_probs(pr, tab)
    return [SimpleNamespace(probs=pr, valence=v, arousal=a, quality=0.8) for _ in range(n)]


@pytest.mark.parametrize("user, strategy, expr", [
    ("happiness", MIRRORING, "happy"),
    # pure surprise: V ~0.15 (< 0.2), A ~0.72 -> paper's high-arousal rule wins (tunable: micro_loop.*)
    ("surprise", DEESCALATION, "calm"),
    ("sadness", SUPPORT, "empathy"),
    ("anger", DEESCALATION, "calm"),          # high arousal, negative
    ("fear", DEESCALATION, "calm"),
    ("neutral", ATTENTIVE, "listening"),
])
def test_micro_loop_strategy(user, strategy, expr):
    d = decide(frames(user), L, MICRO)
    assert (d.strategy, d.expression) == (strategy, expr)
    assert 0.3 <= d.intensity <= 0.8


def test_micro_loop_needs_enough_frames():
    assert decide(frames("happiness", n=2), L, MICRO) is None


# ---------------------------------------------------------------- .ferpack extraction
def test_ferpack_zip_extraction(tmp_path):
    pack = tmp_path / "m.ferpack"
    with zipfile.ZipFile(pack, "w") as z:
        z.writestr("fer.json", "{}")
        z.writestr("model.onnx", b"x")
    _safe_extract(pack, tmp_path / "out")
    assert (tmp_path / "out" / "fer.json").exists()
    bad = tmp_path / "bad.ferpack"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError):
        _safe_extract(bad, tmp_path / "out2")
