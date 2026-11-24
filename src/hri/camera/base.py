from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.width = int(cfg.get("width", 640))
        self.height = int(cfg.get("height", 480))
        self.flip = bool(cfg.get("flip_horizontal", False))

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def _read(self) -> np.ndarray | None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def read(self) -> np.ndarray | None:
        """Latest frame as BGR uint8 (H, W, 3), or None."""
        frame = self._read()
        if frame is not None and self.flip:
            frame = frame[:, ::-1].copy()
        return frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
