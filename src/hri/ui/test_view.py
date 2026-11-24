"""`hri test`: live FER debug window (no microphone, no LLM, no TTS).

    +-----------------------------+---------------------------+
    |  camera (mirrored)          |  raw / calibrated label   |
    |  + YuNet box + landmarks    |  8 class bars (raw tick)  |
    |  + label                    |  valence-arousal plot     |
    |                             |  ES (5 s), trend          |
    |                             |  micro-loop strategy      |
    |                             |  robot face (live)        |
    +-----------------------------+---------------------------+
Keys: q/Esc quit, c re-measure baseline, s screenshot.
Same pipeline as `hri run`: YuNet -> ResNet-18 -> EMA + baseline -> AffectAggregator -> micro-loop.
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
STRATEGY_COLORS = {"mirroring": "#eb6834", "support": "#eda100", "de-escalation": "#6250d6",
                   "attentive": "#2a78d6"}
BG, PANEL, TEXT, DIM = (14, 14, 20), (24, 24, 32), (230, 230, 235), (140, 140, 150)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def run_test(cfg, seconds: float = 0.0, no_calib: bool = False, csv_path: str | None = None,
             screenshot: str | None = None) -> int:
    from hri.affect.aggregator import AffectAggregator
    from hri.camera import create_camera
    from hri.dialogue.micro_loop import decide
    from hri.models import ModelRegistry
    from hri.ui.display import configure_sdl, open_display
    from hri.ui.expressions import FaceController
    from hri.ui.face_renderer import FaceRenderer
    from hri.vision.face_detector import FaceDetector
    from hri.vision.fer import FERModel, face_quality, square_crop
    from hri.vision.postproc import FERStream

    configure_sdl(cfg.section("ui"))   # kmsdrm on Pi OS Lite (no desktop)
    import pygame as pg

    reg = ModelRegistry(cfg)
    v = cfg.section("vision")
    dcfg = v["face_detector"]
    det = FaceDetector(reg.require(dcfg["model"]), dcfg["score_threshold"], dcfg["min_face_size"])
    fer = FERModel(reg.require(v["fer"]["model"]), int(v["fer"].get("num_threads", 2)))
    labels = fer.labels
    stream = FERStream(labels, v)
    agg = AffectAggregator(cfg.section("affect"), labels)
    mcfg = cfg.section("micro_loop")
    margin = float(dcfg.get("margin", 0.15))

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
    robot = FaceRenderer({"show_debug": False, "background": cfg.get("ui.background", (10, 10, 18)),
                          "face_color": cfg.get("ui.face_color", (80, 220, 255))}, ctrl, surface=face_surf)
    f_big, f_mid, f_small = pg.font.Font(None, 30), pg.font.Font(None, 24), pg.font.Font(None, 20)

    writer, csv_file = None, None
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["t", "face", "raw_label", "label", "conf", "valence", "arousal", "calibrated",
                         *[f"raw_{c}" for c in labels], *[f"p_{c}" for c in labels], "strategy", "infer_ms"])

    if not no_calib:
        stream.start_calibration()
    clock = pg.time.Clock()
    t_start = time.time()
    fps_hist: deque[float] = deque(maxlen=30)
    va_trail: deque[tuple[float, float, float]] = deque()
    last_decision_t, decision, summary, last_summary_t = 0.0, None, {}, 0.0
    af, box, lms, infer_ms, frame = None, None, None, 0.0, None
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
            faces = det.detect(frame)
            crop = square_crop(frame, faces[0].box, margin) if faces else None
            if crop is not None and crop.size:
                t1 = time.perf_counter()
                probs = fer.predict_bgr(crop)
                infer_ms = (time.perf_counter() - t1) * 1000
                msg = {"type": "fer", "t": now, "probs": probs,
                       "quality": face_quality(crop, faces[0].score, frame.shape[0])}
                box, lms = faces[0].box, faces[0].landmarks
            else:
                msg, box, lms = {"type": "no_face", "t": now}, None, None
            af = stream.update(msg)
            fps_hist.append(time.perf_counter() - t0)
            if af is not None and not stream.calibrating:
                agg.add(af.to_msg(labels))
                va_trail.append((now, af.valence, af.arousal))
            while va_trail and va_trail[0][0] < now - 2.0:
                va_trail.popleft()

            if not stream.calibrating and now - last_decision_t >= float(mcfg.get("update_sec", 1.0)):
                last_decision_t = now
                d = decide(agg.frames_between(now - float(mcfg.get("window_sec", 1.0)), now), labels, mcfg)
                if d is not None:
                    decision = d
                    ctrl.set_expression(d.expression, d.intensity)
            if now - last_summary_t >= 0.5:
                last_summary_t = now
                summary = agg.summarize(now - 5.0, now)
            if writer is not None:
                writer.writerow([round(now, 3), af is not None,
                                 labels[int(af.raw.argmax())] if af else "", af.label if af else "",
                                 round(af.confidence, 4) if af else "", round(af.valence, 4) if af else "",
                                 round(af.arousal, 4) if af else "", af.calibrated if af else "",
                                 *([round(float(x), 4) for x in af.raw] if af else [""] * len(labels)),
                                 *([round(float(x), 4) for x in af.probs] if af else [""] * len(labels)),
                                 decision.strategy if decision else "", round(infer_ms, 1)])

            # ---------------- draw: camera
            screen.fill(BG)
            fh, fw = frame.shape[:2]
            scale = min(cam_w / fw, cam_h / fh)
            sw, sh = int(fw * scale), int(fh * scale)
            ox, oy = (cam_w - sw) // 2, (cam_h - sh) // 2
            rgb = np.ascontiguousarray(frame[..., ::-1])
            surf = pg.transform.smoothscale(pg.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB"), (sw, sh))
            screen.blit(surf, (ox, oy))
            if box is not None:
                x, y, w, h = box
                col = _rgb(EMOTION_COLORS.get(af.label if af else "neutral", "#888888"))
                r = pg.Rect(ox + x * scale, oy + y * scale, w * scale, h * scale)
                pg.draw.rect(screen, col, r, 3)
                for lx, ly in lms:
                    pg.draw.circle(screen, (255, 255, 255), (ox + float(lx) * scale, oy + float(ly) * scale), 2)
                if af is not None:
                    tag = f"{af.label} {af.confidence:.0%}"
                    ts = f_mid.render(tag, True, (255, 255, 255))
                    ty = r.y - 26 if r.y - 26 >= oy else r.y + 2      # inside the box if no room above
                    tx = max(ox, r.x)
                    pg.draw.rect(screen, col, (tx, ty, ts.get_width() + 10, 26))
                    screen.blit(ts, (tx + 5, ty + 4))
            else:
                ts = f_big.render("no face", True, (255, 120, 120))
                screen.blit(ts, (ox + 12, oy + 12))
            if stream.calibrating:
                prog = stream.calibrator.progress
                bar = pg.Rect(ox + 20, oy + sh - 60, sw - 40, 34)
                pg.draw.rect(screen, (0, 0, 0), bar)
                pg.draw.rect(screen, (255, 160, 0), (bar.x, bar.y, bar.w * prog, bar.h))
                txt = f"CALIBRATING baseline - keep a resting face  {prog:.0%}"
                screen.blit(f_mid.render(txt, True, (255, 255, 255)), (bar.x + 10, bar.y + 8))

            # ---------------- draw: panel
            px = cam_w
            pg.draw.rect(screen, PANEL, (px, 0, pw, H))
            yy = 10
            fps = len(fps_hist) / max(1e-6, sum(fps_hist))
            head = f"FER {infer_ms:4.0f} ms | {fps:4.1f} FPS"
            screen.blit(f_small.render(head if compact else f"{fer.name} | {head}", True, DIM), (px + 10, yy))
            yy += 22
            base = "calibrating" if stream.calibrating else (
                "baseline ON" if stream.calibrator.is_calibrated else "baseline OFF")
            keys = "[c]alib [s]hot [q]uit" if compact else "[c] recalibrate  [s] screenshot  [q] quit"
            screen.blit(f_small.render(f"{base}  {keys}", True, DIM), (px + 10, yy))
            yy += 28
            if af is not None:
                raw_lbl = labels[int(af.raw.argmax())]
                screen.blit(f_mid.render(f"raw {raw_lbl}  ->  {af.label} {af.confidence:.2f}", True, TEXT),
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

            # valence-arousal plot
            size = min(130, H - yy - face_h - 66)      # leave room for the two micro-loop lines
            if size >= 60:
                cx, cy = px + 10 + size // 2, yy + size // 2
                pg.draw.rect(screen, (40, 40, 50), (px + 10, yy, size, size))
                pg.draw.line(screen, DIM, (px + 10, cy), (px + 10 + size, cy))
                pg.draw.line(screen, DIM, (cx, yy), (cx, yy + size))
                for k, (_, tv, ta) in enumerate(va_trail):
                    a = 60 + int(150 * k / max(1, len(va_trail)))
                    pg.draw.circle(screen, (a, a, 255), (cx + tv * size / 2, cy - ta * size / 2), 2)
                if af is not None:
                    pg.draw.circle(screen, (255, 230, 80), (cx + af.valence * size / 2, cy - af.arousal * size / 2), 6)
                screen.blit(f_small.render("V", True, DIM), (px + 12 + size - 12, cy + 2))
                screen.blit(f_small.render("A", True, DIM), (cx + 4, yy + 2))
                tx, step = px + 24 + size, (size - 16) / 3     # 4 lines fit inside the plot height
                if af is not None:
                    screen.blit(f_mid.render(f"V {af.valence:+.2f}", True, TEXT), (tx, yy))
                    screen.blit(f_mid.render(f"A {af.arousal:+.2f}", True, TEXT), (tx, yy + step))
                es = summary.get("stability") if summary.get("face_visible") else None
                screen.blit(f_mid.render(f"ES(5s) {es:.2f}" if es is not None else "ES(5s)  -", True, TEXT),
                            (tx, yy + 2 * step))
                screen.blit(f_small.render(f"trend {summary.get('trend', '-')}", True, DIM), (tx, yy + 3 * step))
                yy += size + 10

            if decision is not None:
                col = _rgb(STRATEGY_COLORS.get(decision.strategy, "#888888"))
                screen.blit(f_mid.render(f"micro-loop: {decision.strategy}", True, col), (px + 10, yy))
                screen.blit(f_small.render(f"robot -> {decision.expression} {decision.intensity:.2f}  "
                                           f"(user {decision.dominant})", True, DIM), (px + 10, yy + 22))
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
