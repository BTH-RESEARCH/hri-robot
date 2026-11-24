from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Utterance:
    text: str
    t_start: float
    t_end: float
    samples: np.ndarray | None = None   # 16 kHz float32, for speaker ID
    speaker: str | None = None
