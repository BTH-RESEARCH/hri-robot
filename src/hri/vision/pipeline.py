"""camera frame -> YuNet -> child's face in the central zone -> ResNet-18. Shared by the robot and `hri test`."""
from __future__ import annotations

import time

import numpy as np

from hri.vision.face_detector import FaceDetector
from hri.vision.face_tracker import CentralFaceTracker
from hri.vision.fer import FERModel, face_quality, square_crop


class ChildFER:
    def __init__(self, cfg, reg):
        v = cfg.section("vision")
        d = v["face_detector"]
        self.detector = FaceDetector(reg.require(d["model"]), d["score_threshold"], d["min_face_size"])
        self.model = FERModel(reg.require(v["fer"]["model"]), int(v["fer"].get("num_threads", 2)))
        self.labels = self.model.labels
        self.margin = float(d.get("margin", 0.15))
        self.tracker = CentralFaceTracker(v.get("child_zone"))

    def process(self, frame: np.ndarray, t: float) -> tuple[dict, list, object]:
        """-> (message for FERStream / ChildQueue, all faces, the followed face or None)"""
        h, w = frame.shape[:2]
        faces = self.detector.detect(frame)
        face, track = self.tracker.select(faces, w, h, t)
        crop = square_crop(frame, face.box, self.margin) if face is not None else None
        if crop is None or not crop.size:
            return {"type": "no_face", "t": t, "n_faces": len(faces)}, faces, None
        t1 = time.perf_counter()
        probs = self.model.predict_bgr(crop)
        return {"type": "fer", "t": t, "track": track, "probs": probs, "n_faces": len(faces),
                "quality": face_quality(crop, face.score, h), "box": face.box,
                "infer_ms": (time.perf_counter() - t1) * 1000}, faces, face
