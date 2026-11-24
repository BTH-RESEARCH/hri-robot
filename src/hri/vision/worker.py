"""Vision runs in its own PROCESS so FER never blocks audio / UI (important on Pi5's 4 cores).

The process only does camera -> YuNet -> square crop -> ResNet-18 and sends RAW class probabilities.
Temporal post-processing (EMA + personal baseline) runs in the main process (vision/postproc.py),
so it can be reset / re-calibrated by the dialogue logic.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time

log = logging.getLogger(__name__)


def vision_loop(cfg_data: dict, profile: str, root: str, out_q: mp.Queue, stop: mp.Event) -> None:
    """Process entry point (top-level so it pickles under Windows 'spawn')."""
    from pathlib import Path

    from hri.camera import create_camera
    from hri.config import Config
    from hri.models import ModelRegistry
    from hri.vision.face_detector import FaceDetector
    from hri.vision.fer import FERModel, face_quality, square_crop

    cfg = Config(cfg_data, profile, Path(root))
    reg = ModelRegistry(cfg)
    vcfg = cfg.section("vision")
    det_cfg = vcfg["face_detector"]
    detector = FaceDetector(reg.require(det_cfg["model"]), det_cfg["score_threshold"], det_cfg["min_face_size"])
    fer = FERModel(reg.require(vcfg["fer"]["model"]), int(vcfg["fer"].get("num_threads", 2)))
    margin = float(det_cfg.get("margin", 0.15))
    period = 1.0 / float(vcfg.get("target_fps", 10))
    _put_latest(out_q, {"type": "ready", "t": time.time(), "labels": fer.labels, "model": fer.name})

    with create_camera(cfg.section("camera")) as cam:
        while not stop.is_set():
            t0 = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            faces = detector.detect(frame)
            crop = square_crop(frame, faces[0].box, margin) if faces else None
            if crop is not None and crop.size:
                f = faces[0]
                t1 = time.perf_counter()
                probs = fer.predict_bgr(crop)
                _put_latest(out_q, {
                    "type": "fer", "t": t0, "probs": probs.tolist(),
                    "quality": face_quality(crop, f.score, frame.shape[0]), "det_score": f.score,
                    "box": f.box, "frame_size": frame.shape[1::-1],
                    "infer_ms": (time.perf_counter() - t1) * 1000, "loop_ms": (time.time() - t0) * 1000,
                })
            else:
                _put_latest(out_q, {"type": "no_face", "t": t0})
            time.sleep(max(0.0, period - (time.time() - t0)))


def _put_latest(q: mp.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:  # consumer is slow: drop oldest
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


class VisionProcess:
    def __init__(self, cfg):
        ctx = mp.get_context("spawn")  # same behaviour on Windows / Linux
        self.queue: mp.Queue = ctx.Queue(maxsize=64)
        self.stop_event = ctx.Event()
        self.proc = ctx.Process(target=vision_loop, name="hri-vision", daemon=True,
                                args=(cfg.data, cfg.profile, str(cfg.root), self.queue, self.stop_event))

    def start(self):
        self.proc.start()

    def stop(self):
        self.stop_event.set()
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
