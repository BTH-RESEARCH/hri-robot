"""Camera abstraction: picamera2 (Pi CSI) or OpenCV (laptop webcam). Frames are BGR uint8."""
from __future__ import annotations

from hri.camera.base import Camera


def create_camera(cam_cfg: dict) -> Camera:
    backend = cam_cfg.get("backend", "opencv")
    if backend == "picamera2":
        from hri.camera.picam import PiCamera
        return PiCamera(cam_cfg)
    if backend == "opencv":
        from hri.camera.opencv_cam import OpenCVCamera
        return OpenCVCamera(cam_cfg)
    if backend == "file":
        from hri.camera.file_cam import FileCamera
        return FileCamera(cam_cfg)
    raise ValueError(f"Unknown camera backend {backend!r}")


__all__ = ["Camera", "create_camera"]
