"""Speaker identification (sherpa-onnx embeddings). Unknown voices get auto-enrolled as guest_N.
Embeddings are persisted in <data_dir>/speakers.npz so users are recognised across sessions."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hri.models import ModelRegistry

log = logging.getLogger(__name__)


class SpeakerID:
    def __init__(self, reg: ModelRegistry, sid_cfg: dict, data_dir: Path, sample_rate: int = 16000):
        import sherpa_onnx

        conf = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(reg.require(sid_cfg["model"])), num_threads=int(sid_cfg.get("num_threads", 1)))
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(conf)
        self.sr = sample_rate
        self.threshold = float(sid_cfg.get("threshold", 0.6))
        self.db_path = data_dir / "speakers.npz"
        self.db: dict[str, np.ndarray] = {}
        if self.db_path.exists():
            self.db = dict(np.load(self.db_path))

    def embed(self, samples: np.ndarray) -> np.ndarray:
        s = self.extractor.create_stream()
        s.accept_waveform(self.sr, samples)
        s.input_finished()
        e = np.asarray(self.extractor.compute(s), np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def enroll(self, name: str, samples: np.ndarray) -> None:
        e = self.embed(samples)
        self.db[name] = (self.db[name] + e) / 2 if name in self.db else e
        self.save()

    def identify(self, samples: np.ndarray, auto_enroll: bool = True) -> str | None:
        if len(samples) < self.sr * 0.8:   # too short for a reliable embedding
            return None
        e = self.embed(samples)
        best, score = None, -1.0
        for name, ref in self.db.items():
            sc = float(e @ ref / (np.linalg.norm(ref) + 1e-9))
            if sc > score:
                best, score = name, sc
        if best is not None and score >= self.threshold:
            return best
        if auto_enroll:
            name = f"guest_{len(self.db) + 1}"
            self.db[name] = e
            self.save()
            log.info("New speaker enrolled: %s", name)
            return name
        return None

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.db_path, **self.db)
