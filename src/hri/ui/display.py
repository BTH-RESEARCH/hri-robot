"""Open a pygame window on a desktop (laptop, Pi OS Desktop) or directly on the screen (Pi OS Lite).

Pi OS Lite has no X11/Wayland: SDL must draw through DRM/KMS ("kmsdrm" video driver).
The pygame wheel from pip bundles an SDL built WITHOUT kmsdrm (only x11/wayland/offscreen), so on the
Pi we use Debian's python3-pygame (system SDL, kmsdrm enabled) through the --system-site-packages venv.
"""
from __future__ import annotations

import os
import sys

FULLSCREEN_ONLY_DRIVERS = {"kmsdrm", "fbcon", "directfb"}


def choose_video_driver(ui_cfg: dict, environ=os.environ, platform: str = sys.platform) -> str | None:
    """auto: kmsdrm on Linux without a desktop session, SDL's default otherwise.
    An explicit SDL_VIDEODRIVER in the environment always wins."""
    if environ.get("SDL_VIDEODRIVER"):
        return environ["SDL_VIDEODRIVER"]
    drv = ui_cfg.get("sdl_videodriver") or "auto"
    if drv == "auto":
        headless = platform.startswith("linux") and not environ.get("DISPLAY") \
            and not environ.get("WAYLAND_DISPLAY")
        return "kmsdrm" if headless else None
    return drv


def configure_sdl(ui_cfg: dict) -> str | None:
    """Must be called before `import pygame` / pygame.init()."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    drv = choose_video_driver(ui_cfg)
    if drv:
        os.environ["SDL_VIDEODRIVER"] = drv
    return drv


def init_pygame() -> None:
    """Display and fonts only. pygame.init() would also start pygame.mixer, which grabs the sound card: on the
    Pi the HDMI speakers accept one client, so TTS then fails with 'Device unavailable [-9985]'."""
    import pygame as pg

    pg.display.init()
    pg.font.init()


def fullscreen_only() -> bool:
    return os.environ.get("SDL_VIDEODRIVER", "") in FULLSCREEN_ONLY_DRIVERS


def open_display(size: tuple[int, int], fullscreen: bool, caption: str):
    """pygame.display.set_mode with Pi-friendly defaults and an actionable error message."""
    import pygame as pg

    init_pygame()
    if fullscreen or fullscreen_only():
        size, flags = (0, 0), pg.FULLSCREEN            # native resolution of the LCD
    else:
        flags = 0
    try:
        screen = pg.display.set_mode(size, flags)
    except pg.error as e:
        raise RuntimeError(display_help(str(e))) from e
    pg.display.set_caption(caption)
    if flags & pg.FULLSCREEN:
        pg.mouse.set_visible(False)
    return screen


def display_help(error: str = "") -> str:
    drv = os.environ.get("SDL_VIDEODRIVER") or "(SDL default)"
    lines = [f"Cannot open the display with SDL video driver {drv}: {error}"]
    if drv == "kmsdrm":
        lines += [
            "Pi OS Lite checklist:",
            "  1. pygame must come from apt (pip's pygame has no kmsdrm):",
            "       sudo apt install python3-pygame && .venv/bin/pip uninstall -y pygame",
            "       python -c 'import pygame; print(pygame.__file__)'   # must be under /usr/lib/python3",
            "  2. user in groups video, render, input:  sudo usermod -aG video,render,input $USER  (log in again)",
            "  3. no other program owns the screen (no X / desktop running on it)",
            "  4. diagnose:  hri check display",
            "  Plan B: run inside a kiosk Wayland compositor:  sudo apt install cage && cage -- .venv/bin/hri run",
        ]
    return "\n".join(lines)


def diagnostics() -> dict:
    """Facts that decide whether the screen can be opened (hri check display)."""
    info = {"platform": sys.platform, "DISPLAY": os.environ.get("DISPLAY"),
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER")}
    try:
        import pygame as pg
        info.update(pygame=pg.version.ver, sdl=".".join(map(str, pg.get_sdl_version())), pygame_path=pg.__file__)
    except Exception as e:  # noqa: BLE001
        info["pygame"] = f"import failed: {e}"
    if sys.platform.startswith("linux"):
        try:
            import grp  # POSIX only
            info["groups"] = sorted(grp.getgrgid(g).gr_name for g in os.getgroups())
        except Exception:  # noqa: BLE001
            pass
        try:
            info["dri"] = sorted(os.listdir("/dev/dri"))
        except OSError:
            info["dri"] = "missing /dev/dri"
    return info
