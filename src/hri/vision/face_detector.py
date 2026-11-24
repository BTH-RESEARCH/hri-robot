"""YuNet face detector (OpenCV DNN, ~1 ms on Pi5 at 320x240)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Face:
    box: tuple[int, int, int, int]   # x, y, w, h
    score: float
    landmarks: np.ndarray            # (5, 2): r-eye, l-eye, nose, r-mouth, l-mouth


class FaceDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.8, min_face_size: int = 40):
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 50)
        self.min_size = min_face_size
        self._size = (0, 0)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        if (w, h) != self._size:
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(bgr)
        out: list[Face] = []
        for f in faces if faces is not None else []:
            x, y, fw, fh = (int(v) for v in f[:4])
            if min(fw, fh) < self.min_size:
                continue
            out.append(Face((x, y, fw, fh), float(f[14]), f[4:14].reshape(5, 2)))
        out.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)  # largest (closest) first
        return out
