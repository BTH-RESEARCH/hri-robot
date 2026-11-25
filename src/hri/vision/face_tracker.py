"""Pick ONE face: the child in the central zone, and keep following it.

The zone does not prove who it is. The track id tells downstream code when a different face took over, so that
the new one gets its own baseline instead of inheriting the child's."""
from __future__ import annotations

import math


class CentralFaceTracker:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.zone_w, self.zone_h = float(cfg.get("width", 0.5)), float(cfg.get("height", 0.9))
        self.keep_sec = float(cfg.get("keep_sec", 10.0))      # looked away and came back: same child
        self.switch_sec = float(cfg.get("switch_sec", 2.0))   # another face must stay this long to take over
        self.track = 0
        self.last = None            # last box of the followed face
        self.last_t = -1e9
        self._other_since: float | None = None

    def zone(self, w: int, h: int) -> tuple[int, int, int, int]:
        zw, zh = int(w * self.zone_w), int(h * self.zone_h)
        return (w - zw) // 2, (h - zh) // 2, zw, zh

    @staticmethod
    def _centre(b) -> tuple[float, float]:
        return b[0] + b[2] / 2, b[1] + b[3] / 2

    def _same(self, box) -> bool:
        (x, y), (u, v) = self._centre(box), self._centre(self.last)
        size = max(self.last[2], self.last[3])
        return math.hypot(x - u, y - v) < 0.8 * size and 0.5 < max(box[2], box[3]) / size < 2.0

    def select(self, faces: list, w: int, h: int, t: float):
        """-> (face or None, track id). faces need .box = (x, y, w, h)."""
        zx, zy, zw, zh = self.zone(w, h)
        inside = [f for f in faces if zx <= self._centre(f.box)[0] <= zx + zw
                  and zy <= self._centre(f.box)[1] <= zy + zh]

        def rank(f):   # close to the middle, slightly preferring larger faces
            cx, cy = self._centre(f.box)
            return math.hypot(cx - w / 2, cy - h / 2) / w - 0.5 * max(f.box[2], f.box[3]) / w

        if self.last is not None and t - self.last_t > self.keep_sec:
            self.last = None
        if self.last is not None:
            same = [f for f in inside if self._same(f.box)]
            if same:
                u, v = self._centre(self.last)
                f = min(same, key=lambda f: math.hypot(self._centre(f.box)[0] - u, self._centre(f.box)[1] - v))
                self.last, self.last_t, self._other_since = f.box, t, None
                return f, self.track
            if not inside:
                self._other_since = None
                return None, self.track
            if self._other_since is None:
                self._other_since = t
            if t - self._other_since < self.switch_sec:
                return None, self.track
        if not inside:
            return None, self.track
        f = min(inside, key=rank)
        self.track += 1
        self.last, self.last_t, self._other_since = f.box, t, None
        return f, self.track
