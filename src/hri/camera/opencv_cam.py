from __future__ import annotations

import sys

import cv2
import numpy as np

from hri.camera.base import Camera


class OpenCVCamera(Camera):
    def start(self) -> None:
        idx = int(self.cfg.get("index", 0))
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # DSHOW opens much faster on Windows
        self.cap = cv2.VideoCapture(idx, api)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, int(self.cfg.get("fps", 30)))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {idx}")

    def _read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
