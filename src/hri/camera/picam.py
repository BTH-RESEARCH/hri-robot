"""Raspberry Pi CSI camera via picamera2.

picamera2 is installed with apt (python3-picamera2), so the venv must be created with
`python3 -m venv --system-site-packages .venv` (scripts/setup_pi.sh does this).
"""
from __future__ import annotations

import numpy as np

from hri.camera.base import Camera


class PiCamera(Camera):
    def start(self) -> None:
        from picamera2 import Picamera2  # import here: not available on laptops

        self.cam = Picamera2()
        # "RGB888" in libcamera is stored B,G,R in memory -> already BGR for OpenCV
        conf = self.cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": float(self.cfg.get("fps", 30))},
            buffer_count=2,
        )
        self.cam.configure(conf)
        self.cam.start()

    def _read(self) -> np.ndarray | None:
        return self.cam.capture_array("main")

    def stop(self) -> None:
        if getattr(self, "cam", None) is not None:
            self.cam.stop()
            self.cam.close()
