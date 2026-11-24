from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Utterance:
    text: str
    t_start: float
    t_end: float
