import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hri.dialogue.micro_loop import ATTENTIVE, DEESCALATION, MIRRORING, SUPPORT, decide
from hri.models.registry import _safe_extract
from hri.vision.fer import FERPLUS_CLASSES, preprocess_rgb, square_crop, va_from_probs, va_table
from hri.vision.postproc import BaselineCalibrator, EMASmoother, FERStream

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


# ---------------------------------------------------------------- EMA + baseline (ported)
def test_ema_and_reset():
    s = EMASmoother(alpha=0.3, reset_after=2)
    a, b = onehot("neutral"), onehot("happiness")
    s.update(a)
    out = s.update(b)
    assert np.allclose(out, 0.3 * b + 0.7 * a)
    for _ in range(3):
        s.update(None)
    assert s.ema is None


def test_baseline_calibration_matches_demo_formula():
    cal = BaselineCalibrator(L, calib_seconds=1.0, min_samples=5)
    rest = onehot("neutral", 0.6)
    done = [cal.update_calibration(rest, 0.25) for _ in range(5)]
    assert done[-1] and cal.is_calibrated
    p = onehot("sadness", 0.3)
    p[0] = 0.5
    p /= p.sum()
    out = cal.adjust(p)
    # demo formula: neutral p-0.15; above-baseline p + 1.5*delta; else p*0.5; renormalise
    exp = np.where(p - cal.p_base > 0, p + 1.5 * (p - cal.p_base), p * 0.5)
    exp[0] = max(0.0, p[0] - 0.15)
    assert np.allclose(out, exp / exp.sum(), atol=1e-6)
    assert L[int(out.argmax())] == "sadness"     # subtle sadness surfaces over the resting-face neutral


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
