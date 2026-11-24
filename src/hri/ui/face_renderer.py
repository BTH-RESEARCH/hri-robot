"""pygame renderer for the parametric robot face. Must run on the MAIN thread."""
from __future__ import annotations

import math
import random
import threading
import time

from hri.ui.expressions import FaceController, FaceParams


def _bezier(p0, p1, p2, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((x, y))
    return pts


class FaceRenderer:
    def __init__(self, ui_cfg: dict, controller: FaceController, surface=None):
        """surface=None: open the robot window (hri run). surface=<pygame.Surface>: draw the face into
        that surface only, e.g. a panel of the `hri test` debug window."""
        from hri.ui.display import configure_sdl, init_pygame, open_display

        configure_sdl(ui_cfg)   # kmsdrm on Pi OS Lite (no desktop)
        import pygame

        self.pg = pygame
        self.cfg = ui_cfg
        self.ctrl = controller
        init_pygame()
        if surface is not None:
            self.screen = surface
        else:
            self.screen = open_display((int(ui_cfg.get("width", 800)), int(ui_cfg.get("height", 480))),
                                       bool(ui_cfg.get("fullscreen")), "HRI Robot")
        self.w, self.h = self.screen.get_size()
        self.bg = tuple(ui_cfg.get("background", (10, 10, 18)))
        self.fg = tuple(ui_cfg.get("face_color", (80, 220, 255)))
        self.fps = int(ui_cfg.get("fps", 30))
        self.font = pygame.font.Font(None, 26)
        self.cur = FaceParams()
        self._next_blink = time.time() + random.uniform(2, 5)
        self._blink_t0 = -1.0
        self._overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.on_key = None   # optional callback(str) for key presses (e.g. 'c' = re-calibrate)

    # ------------------------------------------------------------------
    def run(self, stop: threading.Event | None = None) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        stop = stop or threading.Event()
        while not stop.is_set():
            dt = clock.tick(self.fps) / 1000.0
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key in (pg.K_ESCAPE, pg.K_q)):
                    stop.set()
                elif ev.type == pg.KEYDOWN and self.on_key is not None:
                    self.on_key(ev.unicode)
            self.step(dt)
            pg.display.flip()
        pg.quit()

    def step(self, dt: float) -> None:
        """Advance the tween towards the controller's target and draw one frame."""
        target, level = self.ctrl.snapshot()
        self.cur = self.cur.lerp(target, 1 - math.exp(-dt * 10))   # smooth tween
        self.draw(self.cur, level)

    def _blink_factor(self) -> float:
        now = time.time()
        if self._blink_t0 < 0 and now >= self._next_blink:
            self._blink_t0 = now
        if self._blink_t0 >= 0:
            p = (now - self._blink_t0) / 0.16
            if p >= 1:
                self._blink_t0 = -1
                self._next_blink = now + random.uniform(2.5, 6)
                return 1.0
            return abs(1 - 2 * p)   # 1 -> 0 -> 1
        return 1.0

    def draw(self, p: FaceParams, level: float) -> None:
        pg, w, h = self.pg, self.w, self.h
        self.screen.fill(self.bg)
        self._overlay.fill((0, 0, 0, 0))
        blink = self._blink_factor()
        ew = w * 0.13 * p.eye_scale
        eh_max = h * 0.34 * p.eye_scale
        eh = max(4.0, eh_max * p.eye_open * blink)
        gx, gy = p.pupil_x * ew * 0.25, p.pupil_y * eh_max * 0.15

        for side in (-1, 1):
            cx, cy = w / 2 + side * w * 0.19 + gx, h * 0.42 + gy
            rect = pg.Rect(0, 0, ew, eh)
            rect.center = (cx, cy)
            pg.draw.rect(self.screen, self.fg, rect, border_radius=int(min(ew, eh) / 2))
            if p.eye_happy > 0.02:   # smiling eyes: bg-colored arc pushes lower lid up
                r = ew * 0.9
                lid_y = cy + eh / 2 + r - eh * p.eye_happy * 0.8
                pg.draw.circle(self.screen, self.bg, (cx, lid_y), r)
            if blink > 0.6 and eh > 20:  # eye highlight
                pg.draw.circle(self.screen, self.bg, (cx + ew * 0.2, cy - eh * 0.25), max(3, ew * 0.1))
            # brows
            by = max(h * 0.08, h * 0.42 - eh_max / 2 - h * 0.05 - p.brow_raise * h * 0.06)
            inner = (w / 2 + side * w * 0.11, by - p.brow_tilt * h * 0.05)
            outer = (w / 2 + side * w * 0.27, by + p.brow_tilt * h * 0.03)
            pg.draw.line(self.screen, self.fg, inner, outer, max(4, int(h * 0.018)))
            if p.blush > 0.02:
                pg.draw.ellipse(self._overlay, (255, 120, 170, int(min(1.0, p.blush * 1.6) * 200)),
                                pg.Rect(cx - ew * 0.7, h * 0.62, ew * 1.4, h * 0.06))

        # mouth (lip-sync opens it with TTS loudness)
        mo = max(p.mouth_open, min(1.0, level) * 0.8)
        mw = w * 0.36 * p.mouth_width
        mx, my = w / 2, h * 0.76
        curve = p.mouth_curve * h * 0.09
        left, right = (mx - mw / 2, my - curve * 0.3), (mx + mw / 2, my - curve * 0.3)
        upper = _bezier(left, (mx, my + curve), right)
        thick = max(4, int(h * 0.018))
        if mo > 0.05:
            lower = _bezier(right, (mx, my + curve + mo * h * 0.16), left)
            pg.draw.polygon(self.screen, self.fg, upper + lower)
        else:
            pg.draw.lines(self.screen, self.fg, False, upper, thick)

        self.screen.blit(self._overlay, (0, 0))
        if self.cfg.get("show_debug"):
            info = f"{self.ctrl.name} {self.ctrl.intensity:.1f}  {self.ctrl.debug}"
            self.screen.blit(self.font.render(info, True, (150, 150, 150)), (10, 10))
        if self.ctrl.caption:
            self.screen.blit(self.font.render(self.ctrl.caption[-90:], True, (200, 200, 200)), (10, h - 30))
