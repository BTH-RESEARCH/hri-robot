"""`hri test`: live debug window (no microphone, no LLM, no TTS).

  left : mirrored camera, central zone, followed face (thick box, coloured by label), ignored faces (grey)
  right: raw > calibrated > stable label, 8 class bars, 4 group bars of the last 5 s, micro-loop, robot face
Keys: q/Esc quit, c learn the resting face again, s screenshot.
"""
from __future__ import annotations

import csv
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EMOTION_COLORS = {   # same palette as fer_resnet_demo
    "neutral": "#2a78d6", "happiness": "#eb6834", "surprise": "#1baf7a", "sadness": "#eda100",
    "anger": "#e87ba4", "disgust": "#008300", "fear": "#6250d6", "contempt": "#e34948",
}
GROUP_COLORS = {"neutral_focus": "#2a78d6", "enjoy_curious": "#eb6834", "confused_sad": "#eda100",
                "dislike_distracted": "#e34948"}
BG, PANEL, TEXT, DIM = (14, 14, 20), (24, 24, 32), (230, 230, 235), (140, 140, 150)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def run_test(cfg, seconds: float = 0.0, no_calib: bool = False, csv_path: str | None = None,
             screenshot: str | None = None) -> int:
    from hri.affect.child_state import GROUPS, ChildQueue
    from hri.camera import create_camera
    from hri.dialogue.micro_loop import decide
    from hri.models import ModelRegistry
    from hri.ui.display import configure_sdl, open_display
    from hri.ui.expressions import FaceController
    from hri.ui.face_renderer import FaceRenderer
    from hri.vision.pipeline import ChildFER
    from hri.vision.postproc import FERStream

    configure_sdl(cfg.section("ui"))   # kmsdrm on Pi OS Lite (no desktop)
    import pygame as pg

    fer = ChildFER(cfg, ModelRegistry(cfg))
    labels = fer.labels
    stream = FERStream(labels, cfg.section("vision"))
    child = ChildQueue(labels, cfg.section("child"))
    mcfg = cfg.section("micro_loop")

    # ---------------- window layout
    tcfg = cfg.section("test_ui")
    # kmsdrm (Pi OS Lite) is always fullscreen at the LCD's native size -> layout adapts to it
    screen = open_display((int(tcfg.get("width", 1100)), int(tcfg.get("height", 620))),
                          bool(tcfg.get("fullscreen", False)), "HRI test - FER live")
    W, H = screen.get_size()
    pw = max(300, int(W * 0.36))                  # right panel width
    cam_w, cam_h = W - pw, H
    compact = H < 600 or pw < 380                 # e.g. the Pi's 800x480 LCD
    face_h = int(min(200, H * (0.25 if compact else 0.3)))
    row = 16 if compact else 20
    face_surf = pg.Surface((pw - 20, face_h))
    ctrl = FaceController("listening")
    robot = FaceRenderer({**cfg.section("ui"), "show_debug": False}, ctrl, surface=face_surf)
    f_big, f_mid, f_small = pg.font.Font(None, 30), pg.font.Font(None, 24), pg.font.Font(None, 20)

    writer, csv_file = None, None
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["t", "track", "n_faces", "raw_label", "ema_label", "calib_top", "stable_label", "group",
                         "conf", "calibrated", *[f"raw_{c}" for c in labels], *[f"p_{c}" for c in labels],
                         "robot_expression", "infer_ms"])

    if not no_calib:
        stream.start_calibration()
    clock = pg.time.Clock()
    t_start = time.time()
    fps_hist: deque[float] = deque(maxlen=30)
    last_decision_t, decision, summary, last_summary_t = 0.0, None, {}, 0.0
    af, infer_ms, frame = None, 0.0, None
    running = True

    with create_camera(cfg.section("camera")) as cam:
        while running:
            dt = clock.tick(30) / 1000.0
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key in (pg.K_ESCAPE, pg.K_q)):
                    running = False
                elif ev.type == pg.KEYDOWN and ev.unicode == "c":
                    stream.start_calibration()
                elif ev.type == pg.KEYDOWN and ev.unicode == "s":
                    path = cfg.data_dir / "snapshots" / f"test_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pg.image.save(screen, str(path))
                    log.info("screenshot -> %s", path)

            t0 = time.perf_counter()
            frame = cam.read()
            if frame is None:
                continue
            now = time.time()
            msg, faces, face = fer.process(frame, now)
            infer_ms = msg.get("infer_ms", infer_ms)
            af = stream.update(msg)
            fps_hist.append(time.perf_counter() - t0)
            if af is None:
                child.add_absent(now)
            else:
                group = child.add(af)
            if now - last_decision_t >= float(mcfg.get("update_sec", 1.0)):
                last_decision_t = now
                d = decide(child.summary(now - 1.0, now), mcfg) or decide(child.summary(now - 3.0, now), mcfg)
                if d is not None:
                    decision = d
                    ctrl.set_expression(d.expression, d.intensity)
            if now - last_summary_t >= 0.5:
                last_summary_t = now
                summary = child.summary(now - 5.0, now)
            if writer is not None:
                row_ = [labels[int(af.raw.argmax())], af.ema, af.top, af.label, group,
                        round(af.confidence, 4), af.calibrated, *[round(float(x), 4) for x in af.raw],
                        *[round(float(x), 4) for x in af.probs]] if af else [""] * (7 + 2 * len(labels))
                writer.writerow([round(now, 3), msg.get("track", ""), msg["n_faces"], *row_,
                                 decision.expression if decision else "", round(infer_ms, 1)])

            # ---------------- draw: camera
            screen.fill(BG)
            fh, fw = frame.shape[:2]
            scale = min(cam_w / fw, cam_h / fh)
            sw, sh = int(fw * scale), int(fh * scale)
            ox, oy = (cam_w - sw) // 2, (cam_h - sh) // 2
            rgb = np.ascontiguousarray(frame[..., ::-1])
            surf = pg.transform.smoothscale(pg.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB"), (sw, sh))
            screen.blit(surf, (ox, oy))
            zx, zy, zw, zh = fer.tracker.zone(fw, fh)
            pg.draw.rect(screen, (90, 200, 120), (ox + zx * scale, oy + zy * scale, zw * scale, zh * scale), 1)
            for f in faces:
                x, y, w, h = f.box
                mine = f is face
                col = _rgb(EMOTION_COLORS.get(af.label, "#888888")) if mine and af else (120, 120, 120)
                r = pg.Rect(ox + x * scale, oy + y * scale, w * scale, h * scale)
                pg.draw.rect(screen, col, r, 3 if mine else 1)
                if mine and af:
                    ts = f_mid.render(f"#{msg['track']} {af.label} {af.confidence:.0%}", True, (255, 255, 255))
                    ty = r.y - 26 if r.y - 26 >= oy else r.y + 2      # inside the box if no room above
                    tx = max(ox, r.x)
                    pg.draw.rect(screen, col, (tx, ty, ts.get_width() + 10, 26))
                    screen.blit(ts, (tx + 5, ty + 4))
            if face is None:
                screen.blit(f_big.render("no face in the central zone", True, (255, 120, 120)), (ox + 12, oy + 12))
            if stream.calibrating:
                bar = pg.Rect(ox + 20, oy + sh - 60, sw - 40, 34)
                pg.draw.rect(screen, (0, 0, 0), bar)
                pg.draw.rect(screen, (255, 160, 0), (bar.x, bar.y, bar.w * stream.calibrator.progress, bar.h))
                screen.blit(f_mid.render(f"baseline: keep a resting face {stream.calibrator.progress:.0%}", True,
                                         (255, 255, 255)), (bar.x + 10, bar.y + 8))

            # ---------------- draw: panel
            px = cam_w
            pg.draw.rect(screen, PANEL, (px, 0, pw, H))
            yy = 10
            fps = len(fps_hist) / max(1e-6, sum(fps_hist))
            head = f"FER {infer_ms:4.0f} ms | {fps:4.1f} FPS"
            screen.blit(f_small.render(head if compact else f"{fer.model.name} | {head}", True, DIM), (px + 10, yy))
            yy += 22
            base = "not calibrated" if stream.calibrating else (
                "baseline ON" if stream.calibrator.is_calibrated else "baseline OFF")
            keys = "[c]alib [s]hot [q]uit" if compact else "[c] recalibrate  [s] screenshot  [q] quit"
            screen.blit(f_small.render(f"{base}  {keys}", True, DIM), (px + 10, yy))
            yy += 28
            if af is not None:
                raw_lbl = labels[int(af.raw.argmax())]
                stable = f"{af.label} {af.confidence:.2f}" + (" (rest)" if af.buffer else "")
                screen.blit((f_small if compact or pw < 420 else f_mid).render(f"{raw_lbl} > {af.top} > {stable}", True, TEXT),
                            (px + 10, yy))
            yy += 28
            bar_w = pw - 130
            for i, c in enumerate(labels):
                p_cal = float(af.probs[i]) if af else 0.0
                p_raw = float(af.raw[i]) if af else 0.0
                bh = row - 4
                screen.blit(f_small.render(c, True, TEXT), (px + 10, yy + 1))
                pg.draw.rect(screen, (45, 45, 55), (px + 90, yy, bar_w, bh))
                pg.draw.rect(screen, _rgb(EMOTION_COLORS.get(c, "#888888")), (px + 90, yy, bar_w * p_cal, bh))
                pg.draw.line(screen, (255, 255, 255), (px + 90 + bar_w * p_raw, yy - 1),
                             (px + 90 + bar_w * p_raw, yy + bh + 1), 2)
                screen.blit(f_small.render(f"{p_cal:.2f}", True, DIM), (px + 95 + bar_w, yy + 1))
                yy += row
            if compact:
                yy += 4
            else:
                screen.blit(f_small.render("bar = calibrated, white tick = raw", True, DIM), (px + 10, yy))
                yy += 24

            # four groups over the last 5 s
            for g in GROUPS:
                if yy + 18 > H - face_h - 52:
                    break
                sh_ = summary.get("shares", {}).get(g, 0.0)
                screen.blit(f_small.render(g, True, TEXT), (px + 10, yy + 1))
                pg.draw.rect(screen, (45, 45, 55), (px + 150, yy, pw - 200, 12))
                pg.draw.rect(screen, _rgb(GROUP_COLORS[g]), (px + 150, yy, (pw - 200) * sh_, 12))
                yy += 18
            if not summary.get("visible", True):
                screen.blit(f_small.render(f"5 s: {summary['reason']}", True, DIM), (px + 10, yy))
            if decision is not None:
                col = _rgb(GROUP_COLORS[decision.group])
                screen.blit(f_mid.render(f"{decision.group} ({decision.reason}) > {decision.expression} "
                                         f"{decision.intensity:.2f}", True, col), (px + 10, H - face_h - 34))
            # robot face
            robot.step(dt)
            screen.blit(face_surf, (px + 10, H - face_h - 10))
            pg.display.flip()

            if seconds and time.time() - t_start >= seconds:
                running = False
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            pg.image.save(screen, screenshot)
    if csv_file:
        csv_file.close()
        log.info("frames -> %s", csv_path)
    pg.quit()
    return 0
