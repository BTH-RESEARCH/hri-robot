"""Replay a video file or a folder of images as a camera (offline demos / regression tests).
    hri run --set camera.backend=file --set camera.source=samples/video.mp4
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from hri.camera.base import Camera

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class FileCamera(Camera):
    def start(self) -> None:
        src = Path(str(self.cfg.get("source") or ""))
        if not src.exists():
            raise FileNotFoundError(f"camera.source not found: {src}")
        self.images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else None
        self.cap = None if self.images else cv2.VideoCapture(str(src))
        self.i = 0
        self.period = 1.0 / float(self.cfg.get("fps", 30))
        self.frame_repeat = int(self.cfg.get("frame_repeat", 1))   # images: hold each N frames
        self._next = time.time()

    def _read(self) -> np.ndarray | None:
        time.sleep(max(0.0, self._next - time.time()))   # real-time pacing
        self._next = time.time() + self.period
        if self.images:
            img = cv2.imread(str(self.images[(self.i // self.frame_repeat) % len(self.images)]))
            self.i += 1
        else:
            ok, img = self.cap.read()
            if not ok:                                   # loop the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self.cap.read()
                if not ok:
                    return None
        if img is not None and self.cfg.get("resize", True):
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return img

    def stop(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
