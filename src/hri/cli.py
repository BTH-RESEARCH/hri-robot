"""`hri` command line. Same commands on laptop and Pi; --profile auto-detects the device.

  hri info                          device, profile, model status
  hri models list|download|hash|publish
  hri check camera|vision|mic|asr|tts|llm|face
  hri chat  [--fake-emotion sad] [--no-tts] [--no-ui]   text input, no mic needed
  hri run                           full robot
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from hri import __version__
from hri.config import Config, load_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


# ---------------------------------------------------------------- info / models
def cmd_info(cfg: Config, args) -> int:
    from hri.models import ModelRegistry
    from hri.platform_utils import system_summary

    print(f"hri-robot {__version__}")
    for k, v in system_summary().items():
        print(f"  {k:10s}: {v}")
    print(f"  {'profile':10s}: {cfg.profile}")
    print(f"  {'models':10s}: {cfg.models_dir}")
    print(f"  {'data':10s}: {cfg.data_dir}")
    reg = ModelRegistry(cfg)
    print("Required models:")
    for name in cfg.get("models.required", []):
        print(f"  [{'x' if reg.is_present(name) else ' '}] {name:18s} {reg.path(name)}")
    if args.dump:
        print(cfg.dump())
    return 0


def cmd_models(cfg: Config, args) -> int:
    from hri.models import ModelRegistry
    from hri.models.registry import sha256sum

    reg = ModelRegistry(cfg)
    if args.action == "list":
        for name in reg.manifest:
            mark = "x" if reg.is_present(name) else " "
            req = "*" if name in cfg.get("models.required", []) else " "
            print(f"[{mark}]{req} {name:18s} {reg.url(name)}")
        print("(* = required by this profile)")
    elif args.action == "download":
        names = args.names or cfg.get("models.required", [])
        failed = []
        for n in names:
            try:
                reg.download(n, force=args.force)
            except Exception as e:  # noqa: BLE001
                logging.error("%s: %s", n, e)
                failed.append(n)
        if failed:
            print(f"Failed: {', '.join(failed)}")
            return 1
    elif args.action == "hash":
        for f in args.names:
            print(f"{sha256sum(Path(f))}  {f}")
    elif args.action == "publish":
        if len(args.names) != 2:
            print("usage: hri models publish <manifest-name> <file> [--tag TAG] [--repo OWNER/REPO]")
            return 2
        name, file = args.names
        digest = reg.publish(Path(file), name, tag=args.tag, repo=args.repo)
        print(f"Published {file} as {name}; sha256={digest} written to {reg.manifest_path}")
    return 0


# ---------------------------------------------------------------- hardware checks
def cmd_check(cfg: Config, args) -> int:
    from hri.models import ModelRegistry

    reg = ModelRegistry(cfg)
    what = args.what

    if what == "camera":
        from hri.camera import create_camera
        with create_camera(cfg.section("camera")) as cam:
            n, t0 = 0, time.time()
            while time.time() - t0 < args.seconds:
                f = cam.read()
                n += f is not None
            print(f"camera ok: {n / args.seconds:.1f} fps, frame={None if f is None else f.shape}")

    elif what == "fer":
        # parity: ONNX + our preprocessing on THIS device vs the exporter's reference logits
        import numpy as np

        from hri.vision.fer import FERModel, check_parity
        m = FERModel(reg.require(cfg.get("vision.fer.model")), int(cfg.get("vision.fer.num_threads", 2)))
        print(f"model {m.name}: classes={m.labels}")
        r = check_parity(m, max_images=args.n)
        print(f"parity on {r['n']} test images (ref={r['ref']}): agree={r['agree_pct']:.1f}%  "
              f"acc={r['acc_pct']:.1f}%  max|dlogit|={r['max_abs_logit_diff']:.2e}")
        x = np.zeros((1, 3, int(m.meta["img_size"]), int(m.meta["img_size"])), np.float32)
        t0 = time.perf_counter()
        for _ in range(20):
            m.logits(x)
        print(f"inference: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/face "
              f"({cfg.get('vision.fer.num_threads')} threads)")
        return 0 if r["agree_pct"] >= 99.0 else 1

    elif what == "vision":
        # live text output: camera -> central face -> FER -> post-processing -> 4 groups -> micro-loop
        from hri.affect.child_state import ChildQueue
        from hri.camera import create_camera
        from hri.dialogue.micro_loop import decide
        from hri.vision.pipeline import ChildFER
        from hri.vision.postproc import FERStream
        fer = ChildFER(cfg, reg)
        stream, child = FERStream(fer.labels, cfg.section("vision")), ChildQueue(fer.labels, cfg.section("child"))
        mcfg = cfg.section("micro_loop")
        if not args.no_calib:
            stream.start_calibration()
        n, t_start, last = 0, time.time(), 0.0
        with create_camera(cfg.section("camera")) as cam:
            while time.time() - t_start < args.seconds:
                frame = cam.read()
                if frame is None:
                    continue
                now = time.time()
                msg, faces, _ = fer.process(frame, now)
                af = stream.update(msg)
                ms = (time.time() - now) * 1000
                n += 1
                if af is None:
                    child.add_absent(now)
                    print(f"no face in the zone ({len(faces)} in the image)  {ms:4.0f} ms")
                else:
                    group = child.add(af)
                    print(f"#{msg['track']} raw {fer.labels[int(af.raw.argmax())]:9s} > {af.top:9s} > {af.label:9s} "
                          f"{group:18s} {'' if af.calibrated else 'not calibrated'} {ms:4.0f} ms")
                if now - last >= float(mcfg.get("update_sec", 1.0)):
                    last = now
                    d = decide(child.summary(now - 1.0, now), mcfg)
                    if d:
                        print(f"  >> child {d.group} -> robot '{d.expression}' {d.intensity:.2f}")
        print(f"{n / max(1e-6, time.time() - t_start):.1f} FPS (detector + FER + post-processing)")

    elif what == "mic":
        import numpy as np
        import sounddevice as sd
        print(sd.query_devices())
        sr = int(cfg.get("audio.sample_rate", 16000))
        print(f"Recording {args.seconds}s ...")
        rec = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype="float32",
                     device=cfg.get("audio.input_device"))
        sd.wait()
        print(f"RMS={float(np.sqrt(np.mean(rec ** 2))):.4f} - playing back")
        _play(cfg, rec, sr)

    elif what == "speaker":
        # opens the robot window first, like `hri run`: catches anything in the UI that grabs the sound card
        import numpy as np

        from hri.ui.display import configure_sdl, open_display
        configure_sdl(cfg.section("ui"))
        open_display((640, 400), False, "hri check speaker")
        import pygame as pg
        sr = 22050
        t = np.arange(int(sr * 1.5)) / sr
        try:
            _play(cfg, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        except RuntimeError as e:
            print(e)
            return 1
        finally:
            pg.quit()
        print("OK: a 1.5 s tone was played. If you heard nothing: check the screen's volume and its audio input")

    elif what == "asr":
        import queue

        from hri.audio.asr import ASRWorker
        q: queue.Queue = queue.Queue()
        w = ASRWorker(cfg, reg, q, on_speech_start=lambda t: print("... speech"))
        w.save_dir = args.save
        w.start()
        print(f"Speak now ({args.seconds}s)")
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            try:
                u = q.get(timeout=0.5)
                print(f"-> {u.text}  ({u.t_end - u.t_start:.1f}s audio)")
            except queue.Empty:
                pass
        w.stop()

    elif what == "tts":
        from hri.audio.tts import Speaker
        sp = Speaker(reg, cfg.section("audio"))
        text = args.text or "Xin chào, mình là robot Grace. Rất vui được gặp bạn!"
        t0 = time.perf_counter()
        samples, sr = sp.synth(text)
        dt = time.perf_counter() - t0
        print(f"synth {dt * 1000:.0f} ms for {len(samples) / sr:.2f}s audio (RTF {dt / (len(samples) / sr):.2f})")
        sp.say(text)
        sp.end_turn()

    elif what == "llm":
        from hri.llm.client import LLMClient
        from hri.llm.expression_parser import parse_full
        from hri.llm.prompts import build_system_prompt, build_user_message
        cl = LLMClient(cfg.section("llm"))
        if args.list_models:
            for prov, models in cl.list_models().items():
                print(f"[{prov}]")
                for m in models:
                    print(f"  {m}")
            return 0
        allowed = cfg.get("expression.allowed")
        from hri.dialogue.app import fake_child
        msgs = [{"role": "system", "content": build_system_prompt(allowed)},
                {"role": "user", "content": build_user_message(
                    args.text or "kể cho bé nghe về con mèo", fake_child("happiness"))}]
        t0 = time.perf_counter()
        out = cl.complete(msgs)
        print(f"raw ({(time.perf_counter() - t0) * 1000:.0f} ms): {out}")
        for s in parse_full(out, allowed):
            print(f"  [{s.emotion}|{s.intensity:.1f}] {s.text}")

    elif what == "display":
        # can we draw on the screen? (Pi OS Lite: kmsdrm, needs apt's pygame + video/render/input groups)
        from hri.ui.display import configure_sdl, diagnostics, open_display
        drv = configure_sdl(cfg.section("ui"))
        for k, val in diagnostics().items():
            print(f"  {k:16s}: {val}")
        print(f"  {'chosen driver':16s}: {drv or '(SDL default)'}")
        try:
            screen = open_display((640, 400), False, "hri check display")
        except RuntimeError as e:
            print(e)
            return 1
        import pygame as pg
        w, h = screen.get_size()
        print(f"OK: display opened by '{pg.display.get_driver()}' at {w}x{h} - showing a test pattern for "
              f"{args.seconds:.0f}s")
        font = pg.font.Font(None, max(24, h // 12))
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            for ev in pg.event.get():
                if ev.type == pg.KEYDOWN:
                    print(f"  key: {pg.key.name(ev.key)!r}  (keyboard input works)")
            screen.fill((10, 10, 18))
            pg.draw.rect(screen, (80, 220, 255), (0, 0, w, h), 6)
            pg.draw.circle(screen, (80, 220, 255), (w // 2, h // 2), min(w, h) // 5, 6)
            screen.blit(font.render(f"{pg.display.get_driver()}  {w}x{h}", True, (230, 230, 235)), (20, 20))
            pg.display.flip()
            pg.time.wait(30)
        pg.quit()

    elif what == "face":
        import threading

        from hri.ui.expressions import PRESETS, FaceController
        from hri.ui.face_renderer import FaceRenderer
        ctrl = FaceController()
        stop = threading.Event()

        def cycle():
            while not stop.is_set():
                for name in PRESETS:
                    if stop.is_set():
                        return
                    ctrl.set_expression(name, 1.0)
                    ctrl.set_caption(name)
                    for i in range(20):   # fake talking for a moment
                        ctrl.set_level(abs((i % 6) - 3) / 3 if i < 10 else 0)
                        time.sleep(0.1)
        threading.Thread(target=cycle, daemon=True).start()
        FaceRenderer(cfg.section("ui"), ctrl).run(stop)
    return 0


# ---------------------------------------------------------------- run / chat
def _play(cfg: Config, samples, sr: int) -> None:
    import numpy as np

    from hri.audio.output import Output, convert
    stream, rate, ch = Output(cfg.get("audio.output_device")).open(sr)
    with stream as out:
        out.write(convert(np.concatenate([samples.reshape(-1), np.zeros(sr // 2, np.float32)]), sr, rate, ch))


def cmd_test(cfg: Config, args) -> int:
    from hri.ui.test_view import run_test
    return run_test(cfg, seconds=args.seconds, no_calib=args.no_calib, csv_path=args.csv,
                    screenshot=args.screenshot)


def cmd_chat(cfg: Config, args) -> int:
    from hri.dialogue.app import RobotApp
    RobotApp(cfg, text_mode=True, use_vision=not args.no_vision, use_tts=not args.no_tts,
             use_ui=not args.no_ui, fake_emotion=args.fake_emotion, greet=not args.no_greet).run()
    return 0


def cmd_run(cfg: Config, args) -> int:
    from hri.dialogue.app import RobotApp
    RobotApp(cfg, text_mode=False, use_vision=not args.no_vision, use_tts=True, use_ui=not args.no_ui,
             greet=not args.no_greet).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hri", description="Emotion-aware HRI robot")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-p", "--profile", default="auto", help="auto | laptop | pi5 (default: auto-detect)")
    p.add_argument("-c", "--config", help="extra YAML merged on top of the profile")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    p.add_argument("--log-level", default=None)
    # the same global options are also accepted AFTER the sub-command (hri run --set ui.fullscreen=false)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-p", "--profile", dest="sub_profile", default=None, help=argparse.SUPPRESS)
    common.add_argument("-c", "--config", dest="sub_config", default=None, help=argparse.SUPPRESS)
    common.add_argument("--set", dest="sub_set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config key")
    common.add_argument("--log-level", dest="sub_log_level", default=None, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_parser = sub.add_parser
    sub.add_parser = lambda *a, **kw: _add_parser(*a, parents=[common], **kw)

    s = sub.add_parser("info", help="show device, profile and model status")
    s.add_argument("--dump", action="store_true", help="print the merged config")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("models", help="manage model weights")
    s.add_argument("action", choices=["list", "download", "hash", "publish"])
    s.add_argument("names", nargs="*")
    s.add_argument("--force", action="store_true")
    s.add_argument("--tag")
    s.add_argument("--repo")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("check", help="test one component")
    s.add_argument("what", choices=["display", "camera", "fer", "vision", "speaker", "mic", "asr", "tts", "llm", "face"])
    s.add_argument("--seconds", type=float, default=5.0)
    s.add_argument("--n", type=int, default=None, help="(fer) number of test images, default all")
    s.add_argument("--no-calib", action="store_true", help="(vision) skip the baseline measurement")
    s.add_argument("--text")
    s.add_argument("--save", help="(asr) save every recognised utterance as WAV in this folder")
    s.add_argument("--list-models", action="store_true", help="(llm) list models your API keys can use")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("test", help="live FER debug window: camera + face box + expression results")
    s.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (default: until q)")
    s.add_argument("--no-calib", action="store_true", help="do not learn the resting-face baseline")
    s.add_argument("--csv", help="save per-frame results (raw, EMA, stable label, group, probabilities) to a CSV")
    s.add_argument("--screenshot", help="save the last frame of the window to this PNG")
    s.set_defaults(func=cmd_test)

    for name, func, helptext in (("chat", cmd_chat, "text input instead of mic"), ("run", cmd_run, "full robot")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--no-vision", action="store_true")
        s.add_argument("--no-ui", action="store_true")
        s.add_argument("--no-greet", action="store_true")
        if name == "chat":
            s.add_argument("--no-tts", action="store_true")
            s.add_argument("--fake-emotion", help="pretend the child shows this label or group (no camera needed)")
        s.set_defaults(func=func)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # Vietnamese text on Windows consoles / pipes
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    profile = args.sub_profile or args.profile
    cfg = load_config(profile, args.sub_config or args.config, [*args.set, *args.sub_set])
    _setup_logging(args.sub_log_level or args.log_level or cfg.get("app.log_level", "INFO"))
    try:
        return int(args.func(cfg, args) or 0)
    except FileNotFoundError as e:
        logging.error("%s", e)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
