"""Frame queue of the child's state and the four pedagogical response groups.

The groups are INDIRECT signals: FERPlus has no 'confused' or 'distracted' class. 'Distracted' is only inferred
from a missing face, 'confused' only from sadness / fear.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

FOCUS, ENJOY, CONFUSED, DISLIKE = "neutral_focus", "enjoy_curious", "confused_sad", "dislike_distracted"
GROUPS = [FOCUS, ENJOY, CONFUSED, DISLIKE]
EMOTION_TO_GROUP = {"neutral": FOCUS, "happiness": ENJOY, "surprise": ENJOY, "sadness": CONFUSED,
                    "fear": CONFUSED, "anger": DISLIKE, "disgust": DISLIKE, "contempt": DISLIKE}
GROUP_VI = {FOCUS: "trung tính, tập trung", ENJOY: "thích thú, tò mò", CONFUSED: "bối rối hoặc buồn",
            DISLIKE: "không thích hoặc xao nhãng"}


def group_probs(probs, labels: list[str]) -> np.ndarray:
    out = np.zeros(4)
    for p, c in zip(probs, labels):
        out[GROUPS.index(EMOTION_TO_GROUP[c])] += float(p)
    return out


def child_group(frame, labels: list[str], dislike_min_raw: float) -> str:
    """Group of one frame. The post-processor boosts anger / disgust / contempt strongly (a raw 2% can win), so the
    dislike group additionally needs the RAW model to give these classes at least `dislike_min_raw` together.
    Otherwise the frame counts as neutral_focus. The label itself is not changed."""
    g = EMOTION_TO_GROUP[frame.label]
    if g == DISLIKE and group_probs(frame.raw, labels)[GROUPS.index(DISLIKE)] < dislike_min_raw:
        return FOCUS
    return g


@dataclass
class Entry:
    t: float
    group: str | None          # None = the child's face is not visible
    label: str | None = None
    gp: np.ndarray | None = None
    quality: float = 0.0
    calibrated: bool = False


class ChildQueue:
    def __init__(self, labels: list[str], cfg: dict | None = None):
        cfg = cfg or {}
        self.labels = labels
        self.buffer_sec = float(cfg.get("buffer_sec", 60))
        self.min_frames = int(cfg.get("min_frames", 3))
        self.away_ratio = float(cfg.get("away_ratio", 0.6))
        self.away_min_sec = float(cfg.get("away_min_sec", 1.5))
        self.dislike_min_raw = float(cfg.get("dislike_min_raw", 0.15))
        self._q: deque[Entry] = deque()
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._q.clear()

    def _push(self, e: Entry) -> None:
        with self._lock:
            self._q.append(e)
            while self._q and self._q[0].t < e.t - self.buffer_sec:
                self._q.popleft()

    def add(self, frame) -> str:
        """frame: postproc.AffectFrame -> its group"""
        g = child_group(frame, self.labels, self.dislike_min_raw)
        gp, label = group_probs(frame.probs, self.labels), frame.label
        if g != EMOTION_TO_GROUP[label]:          # rejected dislike: its boosted mass must not drive trends either
            k = GROUPS.index(DISLIKE)
            gp[GROUPS.index(FOCUS)] += gp[k]
            gp[k], label = 0.0, "neutral"
        self._push(Entry(frame.t, g, label, gp, frame.quality, frame.calibrated))
        return g

    def add_absent(self, t: float) -> None:
        self._push(Entry(t, None))

    def window(self, t0: float, t1: float) -> list[Entry]:
        with self._lock:
            return [e for e in self._q if t0 <= e.t <= t1]

    def summary(self, t0: float, t1: float) -> dict:
        """Dominance, stability, average metrics and trend of the groups in a time window."""
        w = self.window(t0, t1)
        seen = [e for e in w if e.group is not None]
        if len(w) < self.min_frames:
            return {"visible": False, "reason": "no_data", "n": len(w)}
        absent = 1 - len(seen) / len(w)
        span = w[-1].t - w[0].t
        if absent >= self.away_ratio and span >= self.away_min_sec:
            return {"visible": False, "reason": "away", "group": DISLIKE, "n": len(w), "absent": round(absent, 2)}
        if len(seen) < self.min_frames:
            return {"visible": False, "reason": "no_data", "n": len(w)}
        shares = {g: sum(e.group == g for e in seen) / len(seen) for g in GROUPS}
        top = max(shares, key=shares.get)
        q = np.array([max(1e-3, e.quality) for e in seen])
        gp = np.stack([e.gp for e in seen])
        half = len(seen) // 2
        delta = gp[half:].mean(0) - gp[:half].mean(0) if half >= 2 else np.zeros(4)
        k = int(np.argmax(delta))
        switches = sum(a.group != b.group for a, b in pairwise(seen))
        labels = [e.label for e in seen if e.group == top]
        return {"visible": True, "group": top, "dominance": round(shares[top], 2),
                "shares": {g: round(s, 2) for g, s in shares.items()},
                "mean_probs": {g: round(float(x), 2) for g, x in zip(GROUPS, (q[:, None] * gp).sum(0) / q.sum())},
                "trend": GROUPS[k] if delta[k] > 0.15 else "stable",
                "switches_per_s": round(switches / max(span, 1e-6), 2),
                "label": max(set(labels), key=labels.count),
                "calibrated": sum(e.calibrated for e in seen) / len(seen) > 0.5,
                "absent": round(absent, 2), "n": len(seen)}

    def sustained(self, groups: set[str], now: float, seconds: float, ratio: float = 0.8) -> bool:
        """The child's visible, calibrated face was in `groups` for (almost) the whole last `seconds`."""
        w = self.window(now - seconds, now)
        if len(w) < max(6, 2 * seconds) or w[0].t > now - 0.8 * seconds or w[-1].t < now - 1.0:
            return False
        return sum(e.group in groups and e.calibrated for e in w if e.group) >= ratio * len(w)


# ----------------------------------------------------------------------------- what the child liked
LIKED, DISLIKED, UNCLEAR = "liked", "disliked", "unclear"


def verdict(s: dict) -> tuple[str, float]:
    """Reaction to one piece of content from a window summary -> (verdict, score -1..1)."""
    if not s.get("visible"):
        return (DISLIKED, -0.5) if s.get("reason") == "away" else (UNCLEAR, 0.0)
    if not s["calibrated"]:
        return UNCLEAR, 0.0
    sh = s["shares"]
    score = sh[ENJOY] - sh[DISLIKE] - 0.5 * sh[CONFUSED] - 0.5 * s["absent"]
    return (LIKED if score >= 0.3 else DISLIKED if score <= -0.3 else UNCLEAR), round(score, 2)


class ContentLog:
    """Topic -> reactions, for the current session only (RAM)."""

    def __init__(self):
        self.items: list[dict] = []

    def reset(self) -> None:
        self.items = []

    def add(self, topic: str, summary: dict) -> dict:
        v, score = verdict(summary)
        item = {"topic": topic or "khác", "verdict": v, "score": score, "summary": summary}
        self.items.append(item)
        del self.items[:-50]
        return item

    def last(self) -> dict | None:
        return self.items[-1] if self.items else None

    def _topics(self, want: str) -> list[str]:
        scores: dict[str, list[float]] = {}
        for i in self.items:
            if i["verdict"] != UNCLEAR and i["topic"] != "khác":
                scores.setdefault(i["topic"], []).append(i["score"])
        mean = {t: float(np.mean(s)) for t, s in scores.items()}
        keep = [t for t, m in mean.items() if (m >= 0.3 if want == LIKED else m <= -0.3)]
        return sorted(keep, key=lambda t: -abs(mean[t]))[:5]

    def likes(self) -> list[str]:
        return self._topics(LIKED)

    def dislikes(self) -> list[str]:
        return self._topics(DISLIKED)
