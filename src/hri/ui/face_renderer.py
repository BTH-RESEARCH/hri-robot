"""pygame port of the robot face designed in faces/index.html. Must run on the MAIN thread.

Drawn natively because Pi OS Lite has no browser. Same geometry (800x480 design, scaled to the screen), lids, hearts,
gleam, glow, scan lines, vignette, blinking, wandering gaze, breathing and boot animation. One addition:
the eyes pulse with the voice, because the design has no mouth.
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections import OrderedDict
from dataclasses import replace

import cv2
import numpy as np

from hri.ui.expressions import FaceController, FaceParams

W, H = 800, 480
EYE_W, EYE_H, EYE_GAP, CY = 162, 212, 100, 240
SS = 2                                  # supersampling of the eye shapes (anti-aliasing)


def _clamp(v, a, b):
    return max(a, min(b, v))


def _quad(p0, p1, p2, n=16):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]


def _cubic(p0, p1, p2, p3, n=24):
    return [tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d
                  for a, b, c, d in zip(p0, p1, p2, p3)) for t in (i / n for i in range(n + 1))]


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
        self.k = min(self.w / W, self.h / H)                      # design px -> screen px
        self.fg = tuple(ui_cfg.get("face_color", (95, 231, 224)))
        self.glow = float(ui_cfg.get("glow", 0.55))
        self.size = float(ui_cfg.get("eye_size", 1.0))
        self.idle = bool(ui_cfg.get("idle_motion", True))
        self.fps = int(ui_cfg.get("fps", 30))
        self.font = pygame.font.Font(None, 26)
        self.cur = FaceParams()
        self.on_key = None   # optional callback(str) for key presses (e.g. 'c' = re-calibrate)
        self._t0: float | None = None
        self._blink_t0, self._blink_dur, self._next_blink = None, 0.15, 1.8
        self._look, self._look_target, self._next_move = [0.0, 0.0], (0.0, 0.0), 2.6
        self._voice = 0.0
        self._bg = self._background()
        self._over = self._overlay(bool(ui_cfg.get("scanlines", False)))
        self._gradient = self._vertical_gradient()
        self._gleam = self._make_gleam()
        # vignette and scan lines only need to be multiplied where the eyes are: elsewhere they are baked in
        self._bg.blit(self._over, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        self._plain_bg = self._background()
        # drawing an eye with its halo is the expensive part; at rest only its position changes
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()

    # ------------------------------------------------------------------ static layers
    def _from_array(self, rgb: np.ndarray):
        return self.pg.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8).swapaxes(0, 1)).convert()

    def _background(self):
        y, x = np.mgrid[0: self.h, 0: self.w]
        d = np.hypot((x - 0.5 * self.w) / (1.2 * self.w), (y - 0.42 * self.h) / (0.9 * self.h))
        stops = [0.0, 0.45, 1.0]
        cols = np.array([[0x0d, 0x15, 0x1d], [0x07, 0x0b, 0x10], [0x02, 0x04, 0x07]], float)
        bg = np.stack([np.interp(d, stops, cols[:, c]) for c in range(3)], -1)
        # top light of the 'glass' (screen blend of a faint white), baked in
        g = 0.05 * np.clip(1 - np.hypot((x - 0.5 * self.w) / (1.3 * self.w), y / (0.8 * self.h)) / 0.6, 0, 1)
        bg = bg + (255 - bg) * g[..., None]
        noise = np.random.default_rng(0).uniform(-0.5, 0.5, bg.shape[:2])[..., None]      # no banding
        return self._from_array(bg + noise + 0.5)

    def _overlay(self, scan: bool):
        """Vignette + scan lines as one multiplicative layer."""
        y, x = np.mgrid[0: self.h, 0: self.w]
        d = np.hypot((x - self.w / 2) / (0.7 * 1.12 * self.w), (y - self.h / 2) / (0.6 * 1.12 * self.h))
        m = 1 - 0.65 * np.clip((d - 0.55) / 0.45, 0, 1)
        if scan:
            m[::4] *= 1 - 0.5 * 0.2
        return self._from_array(np.repeat((255 * m)[..., None], 3, -1))

    def _vertical_gradient(self):
        a = np.interp(np.linspace(0, 1, 64), [0, 0.55, 1], [1, 0.92, 0.72])
        s = self.pg.Surface((1, 64), self.pg.SRCALPHA)
        for i, v in enumerate(a):
            s.set_at((0, i), (255, 255, 255, int(255 * v)))
        return s

    def _make_gleam(self):
        """White reflections (added to RGB only, so they stay inside the eye). Unit square, scaled per eye."""
        n = 128
        y, x = (np.mgrid[0:n, 0:n] + 0.5) / n - 0.5
        g = 0.55 * np.clip(1 - np.hypot(x + 0.22, (y + 0.26) * EYE_H / EYE_W) / 0.34, 0, 1)
        u, v = x + 0.22, y + 0.26
        c, s = math.cos(-0.35), math.sin(-0.35)
        g = np.where(((u * c + v * s) / 0.30) ** 2 + ((-u * s + v * c) / 0.20) ** 2 < 1, g, 0)
        u, v = x - 0.14, (y - 0.12) * EYE_H / EYE_W            # in eye widths, so that the angle is right
        c, s = math.cos(-0.62), math.sin(-0.62)
        streak = ((u * c + v * s) / 0.08) ** 2 + ((-u * s + v * c) / (0.46 * EYE_H / EYE_W)) ** 2 < 1
        g = np.clip(g + 0.13 * streak, 0, 1)[..., None] * (255 - np.array(self.fg, float))
        return self.pg.surfarray.make_surface(g.astype(np.uint8).swapaxes(0, 1))

    # ------------------------------------------------------------------ main loop
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

    def step(self, dt: float, now: float | None = None) -> None:
        """Advance the animation towards the controller's target and draw one frame."""
        dt = min(0.05, max(0.0, dt))
        now = time.monotonic() if now is None else now
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        k = 1 - 0.0015 ** dt                                       # smoothing independent of the frame rate
        target, level = self.ctrl.snapshot()
        asleep = self.ctrl.name == "sleep"
        blink = self._animate(t, asleep)
        self._look = [a + (b - a) * k for a, b in zip(self._look, self._look_target)]
        self.cur = self.cur.lerp(target, k)
        self._voice += (min(1.0, level) - self._voice) * (1 - 0.00001 ** dt)
        voice = round(self._voice * 5) / 5                         # few steps: cached eye shapes
        p = self.cur.lerp(self.cur, 0)
        p.dx += self._look[0] * 58
        p.dy += self._look[1] * 34 + math.sin(t / 1.5) * 2.6 + (math.sin(t / 2.6) * 4 if asleep else 0)
        p.h *= 1 + 0.10 * voice                                    # speaking: the eyes pulse with the voice
        p.scale *= 1 + 0.03 * voice
        boot = 1 - (1 - _clamp(t / 0.62, 0, 1)) ** 3
        self.draw(p, blink, boot, self._look[0], t)

    def _animate(self, t: float, asleep: bool) -> float:
        """Blinking and wandering gaze. Returns how far the eyes are closed by a blink (0..1)."""
        if asleep:
            self._look_target = (0.0, 0.0)
        elif self.idle:
            if t > self._next_blink and self._blink_t0 is None:
                self._blink_t0, self._blink_dur = t, 0.26 if self.ctrl.name == "sleepy" else 0.15
                self._next_blink = t + (0.42 if random.random() < 0.18 else 2.2 + random.random() * 4.2)
            if t > self._next_move:
                a, r = random.random() * math.tau, random.random()
                still = random.random() < 0.3 or self.ctrl.name == "thinking"
                self._look_target = (0.0, 0.0) if still else (math.cos(a) * r * 0.75, math.sin(a) * r * 0.45)
                self._next_move = t + 1.4 + random.random() * 3.0
        if self._blink_t0 is None:
            return 0.0
        e = (t - self._blink_t0) / self._blink_dur
        if e >= 1:
            self._blink_t0 = None
            return 0.0
        return e * 2 if e < 0.5 else (1 - e) * 2

    # ------------------------------------------------------------------ drawing
    def _eye(self, side: int, p: FaceParams, blink: float, boot: float, look_x: float, t: float):
        """-> (eye, halo, centre x, centre y in design px)"""
        sc = p.scale * self.size
        w = EYE_W * p.w * sc * (1 - side * look_x * 0.05)          # the eye further away is a little smaller
        h = EYE_H * p.h * (p.hL if side < 0 else p.hR) * sc * (0.08 + 0.92 * boot) * (0.30 + 0.70 * (1 - blink))
        cx = W / 2 + side * (EYE_W * p.w * sc + EYE_GAP) / 2 + p.dx
        q = 2 / max(self.k, 0.25)                                  # shapes are cached in steps of ~2 screen px
        w, h = max(q, round(w / q) * q), max(q, round(h / q) * q)
        beat = round(math.sin(t / 0.26) * 4) / 4 if p.heart > 0.02 else 0.0
        key = (w, h, round(p.r, 1), round(p.top + blink, 2), round(p.bot, 2), round(side * p.slant) if p.top + blink > 0 else 0,
               round(p.heart, 1), beat, round(p.dim, 1), round(p.tilt * 2) / 2)
        hit = self._cache.get(key)
        if hit is None:
            eye = self._shape(w, h, replace(p, r=key[2], top=key[3], bot=key[4], slant=side * key[5], heart=key[6],
                                            dim=key[8], tilt=key[9]), side, beat)
            hit = self._cache[key] = (eye, self._halo(eye))
            if len(self._cache) > 96:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return *hit, cx, CY + p.dy

    def _shape(self, w: float, h: float, p: FaceParams, side: int, beat: float):
        pg = self.pg
        f = self.k * SS
        pad = 1.3                                                  # hearts are wider than the eye
        sw, sh = max(2, int(w * pad * f)), max(2, int(h * pad * f))
        s = pg.Surface((sw, sh), pg.SRCALPHA)
        ox, oy = sw / 2, sh / 2

        def pts(q):
            return [(ox + x * f, oy + y * f) for x, y in q]

        if p.heart < 0.98:
            r = pg.Rect(0, 0, round(w * f), round(h * f))
            r.center = (round(ox), round(oy))
            pg.draw.rect(s, (*self.fg, int(255 * (1 - p.heart)) if p.heart > 0.02 else 255), r,
                         border_radius=int(min(w, h) * 0.5 * p.r * f))
        if p.heart > 0.02:
            m = 1 + beat * 0.06
            hw, hh = w * 1.10 * m, h * 0.96 * m
            top = -hh * 0.34
            heart = pg.Surface((sw, sh), pg.SRCALPHA)
            pg.draw.polygon(heart, (*self.fg, int(255 * p.heart)), pts(
                _cubic((0, hh * 0.5), (-hw * 0.78, -hh * 0.06), (-hw * 0.52, top - hh * 0.34), (0, top + hh * 0.14))
                + _cubic((0, top + hh * 0.14), (hw * 0.52, top - hh * 0.34), (hw * 0.78, -hh * 0.06), (0, hh * 0.5))))
            s.blit(heart, (0, 0))
        grad = pg.transform.smoothscale(self._gradient, (sw, max(1, round(h * f))))
        full = pg.Surface((sw, sh), pg.SRCALPHA)
        full.fill((255, 255, 255, 255))
        full.fill((255, 255, 255, 184), (0, sh // 2, sw, sh - sh // 2))      # below the eye: last gradient value
        y0 = round(oy - h * f / 2)
        full.fill((0, 0, 0, 0), (0, y0, sw, grad.get_height()))
        full.blit(grad, (0, y0))
        if p.dim < 0.999:
            full.fill((255, 255, 255, int(255 * _clamp(p.dim, 0, 1))), special_flags=pg.BLEND_RGBA_MULT)
        s.blit(full, (0, 0), special_flags=pg.BLEND_RGBA_MULT)

        # lids erase pixels (so the glow follows the visible shape)
        clear = (0, 0, 0, 0)
        big = w * 1.4
        top_amt = _clamp(p.top, 0, 1.2)
        if top_amt > 0.001:
            a = math.radians(-side * p.slant)
            ca, sa = math.cos(a), math.sin(a)
            y1, y2 = -h * 1.6, -h / 2 + h * top_amt + 1
            pg.draw.polygon(s, clear, pts([(x * ca - y * sa, x * sa + y * ca)
                                           for x, y in ((-big / 2, y1), (big / 2, y1), (big / 2, y2), (-big / 2, y2))]))
        if p.bot > 0.001:
            edge, mid = h / 2 - h * p.bot * 1.14, h / 2 - h * p.bot * 0.90
            e = w / 2 * 1.02
            pg.draw.polygon(s, clear, pts([(-big / 2, edge), *_quad((-e, edge), (0, 2 * mid - edge), (e, edge)),
                                           (big / 2, edge), (big / 2, h * 2), (-big / 2, h * 2)]))

        # gleam: a blend towards white inside the eye (adds to RGB only, the eye's alpha clips it)
        gl = pg.transform.smoothscale(self._gleam, (max(1, round(w * f)), max(1, round(h * f))))
        s.blit(gl, (round(ox - w * f / 2), y0), special_flags=pg.BLEND_RGB_ADD)
        if abs(p.tilt) < 0.05:
            return pg.transform.smoothscale(s, (max(1, round(sw / SS)), max(1, round(sh / SS))))
        return pg.transform.rotozoom(s, -p.tilt, 1 / SS)

    def draw(self, p: FaceParams, blink: float = 0.0, boot: float = 1.0, look_x: float = 0.0, t: float = 0.0) -> None:
        pg = self.pg
        self.screen.blit(self._bg, (0, 0))
        ca, sa = math.cos(math.radians(p.tilt)), math.sin(math.radians(p.tilt))
        eyes, box = [], None
        for side in (-1, 1):
            eye, halo, cx, cy = self._eye(side, p, blink, boot, look_x, t)
            dx, dy = cx - W / 2, cy - CY                            # the whole face tilts around its centre
            c = (round(self.w / 2 + (dx * ca - dy * sa) * self.k), round(self.h / 2 + (dx * sa + dy * ca) * self.k))
            r = halo.get_rect(center=c)
            eyes.append((eye, halo, eye.get_rect(center=c), r))
            box = r if box is None else box.union(r)
        box = box.clip(self.screen.get_rect())
        self.screen.blit(self._plain_bg, box, box)
        for _, halo, _, r in eyes:
            self.screen.blit(halo, r, special_flags=pg.BLEND_RGB_ADD)
        for eye, _, r, _ in eyes:
            self.screen.blit(eye, r)
        self.screen.blit(self._over, box, box, special_flags=pg.BLEND_RGB_MULT)
        if self.ctrl.mic_open:
            pg.draw.circle(self.screen, (80, 220, 120), (self.w - 24, 24), 10)
        if self.cfg.get("show_debug"):
            info = f"{self.ctrl.name} {self.ctrl.intensity:.1f}  {self.ctrl.debug}"
            self.screen.blit(self.font.render(info, True, (150, 150, 150)), (10, 10))
        if self.ctrl.caption:
            self.screen.blit(self.font.render(self.ctrl.caption[-90:], True, (200, 200, 200)), (10, self.h - 30))

    def _halo(self, eye):
        """The two CSS drop-shadows of the design (sigma = blur radius), blurred on a 1/8 size alpha mask."""
        pg = self.pg
        g, n = self.glow, 8
        shadows = (((8 + g * 14) * self.k, 0.25 + g * 0.6), ((20 + g * 48) * self.k, g * 0.55))
        m = int(shadows[1][0] * 2.2)
        hw, hh = ((v + 2 * m) // n + 1 for v in eye.get_size())
        base = pg.Surface((hw * n, hh * n))             # black: adding it changes nothing
        if g <= 0.01:
            return base
        a = pg.surfarray.array_alpha(eye)
        pad = np.zeros((hw * n, hh * n), np.uint8)
        x, y = (hw * n - a.shape[0]) // 2, (hh * n - a.shape[1]) // 2
        pad[x:x + a.shape[0], y:y + a.shape[1]] = a
        small = cv2.resize(pad, (hh, hw), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        out = sum(strength * cv2.GaussianBlur(small, (0, 0), max(0.5, radius / n)) for radius, strength in shadows)
        rgb = np.clip(out[..., None] * np.array(self.fg, np.float32), 0, 255).astype(np.uint8)
        pg.surfarray.blit_array(base, cv2.resize(rgb, (hh * n, hw * n), interpolation=cv2.INTER_LINEAR))
        return base
