"""FER model: ResNet-18 FERPlus exported to ONNX, loaded from a .ferpack folder (model.onnx + fer.json).

Preprocessing is read from fer.json and reproduces the training/eval transform exactly
(same code path as fer_resnet_demo / fer-edge):
    face crop (RGB) -> [gray] -> [BOX downsample to train_res] -> RGB -> BILINEAR to img_size
    -> /255 -> (x - mean) / std -> NCHW float32
The model only outputs class logits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

FERPLUS_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def square_crop(img: np.ndarray, box_xywh: tuple, margin: float) -> np.ndarray:
    """Square box around the face, enlarged by `margin` on each side, clipped to the image
    (identical to fer_demo/pipeline.square_crop; YuNet gives x, y, w, h)."""
    x, y, w, h = box_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    H, W = img.shape[:2]
    a, b = int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2)))
    c, d = int(min(W, round(cx + side / 2))), int(min(H, round(cy + side / 2)))
    return img[b:d, a:c]


def load_meta(pack_dir: Path) -> dict:
    meta_path = Path(pack_dir) / "fer.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found - is {pack_dir} an extracted .ferpack?")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def to_train_like(img: Image.Image, meta: dict) -> Image.Image:
    if meta.get("grayscale"):
        img = img.convert("L")
    r = meta.get("train_res")
    if r and min(img.size) > int(r):
        img = img.resize((int(r), int(r)), Image.BOX)
    return img.convert("RGB")


def preprocess_rgb(rgb: np.ndarray | Image.Image, meta: dict) -> np.ndarray:
    """RGB uint8 HxWx3 (or PIL) -> 1x3xSxS float32."""
    img = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    s = int(meta["img_size"])
    img = to_train_like(img, meta).resize((s, s), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.asarray(meta["mean"], np.float32)) / np.asarray(meta["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class FERModel:
    def __init__(self, pack_dir: Path, num_threads: int = 2):
        import onnxruntime as ort

        self.dir = Path(pack_dir)
        self.meta = load_meta(self.dir)
        self.labels: list[str] = list(self.meta["classes"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx = self.dir / self.meta.get("onnx", "model.onnx")
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.name = self.meta.get("name", self.dir.name)
        self.logits(np.zeros((1, 3, int(self.meta["img_size"]), int(self.meta["img_size"])), np.float32))  # warm-up

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0].reshape(-1)

    def predict_rgb(self, rgb: np.ndarray | Image.Image) -> np.ndarray:
        """Face crop (RGB) -> class probabilities (raw, before smoothing / calibration)."""
        return softmax(self.logits(preprocess_rgb(rgb, self.meta)))

    def predict_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.predict_rgb(np.ascontiguousarray(crop_bgr[..., ::-1]))


def face_quality(crop_bgr: np.ndarray, det_score: float, frame_h: int) -> float:
    """0..1 heuristic (detector score, face size, sharpness) - used to weight frames in summaries."""
    import cv2

    size_q = min(1.0, crop_bgr.shape[0] / (0.35 * frame_h))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp_q = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0)
    return float(np.clip(0.4 * det_score + 0.3 * size_q + 0.3 * sharp_q, 0, 1))


def check_parity(model: FERModel, max_images: int | None = None) -> dict:
    """Compare ONNX logits on the pack's testset with the reference logits stored by the exporter.
    Verifies that preprocessing + runtime on THIS device reproduce the training pipeline."""
    import csv

    test_dir = model.dir / "testset"
    ref = np.load(test_dir / "ref_logits.npz")
    variant = model.meta.get("variant", "fp32")
    ref_logits = ref[variant] if variant in ref.files else ref["torch_fp32"]
    with open(test_dir / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: max_images or None]
    diffs, agree, correct = [], 0, 0
    for i, row in enumerate(rows):
        img = Image.open(test_dir / "images" / row["file"]).convert("RGB")
        out = model.logits(preprocess_rgb(img, model.meta))
        diffs.append(float(np.abs(out - ref_logits[i]).max()))
        agree += int(np.argmax(out) == np.argmax(ref_logits[i]))
        correct += int(np.argmax(out) == int(row["label"]))
    n = max(1, len(rows))
    return {"n": len(rows), "ref": variant if variant in ref.files else "torch_fp32",
            "agree_pct": 100.0 * agree / n, "acc_pct": 100.0 * correct / n,
            "max_abs_logit_diff": max(diffs) if diffs else float("nan")}
