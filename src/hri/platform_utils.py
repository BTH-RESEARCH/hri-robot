"""Device detection so the same code runs on a laptop and a Raspberry Pi 5."""
from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def board_model() -> str:
    """Return the board model string on Raspberry Pi, '' elsewhere."""
    p = Path("/proc/device-tree/model")
    try:
        return p.read_bytes().decode(errors="ignore").strip("\x00").strip()
    except OSError:
        return ""


def is_raspberry_pi() -> bool:
    return "raspberry pi" in board_model().lower()


def detect_profile() -> str:
    """Profile name: env HRI_PROFILE, else 'pi5' on a Raspberry Pi, else 'laptop'."""
    env = os.environ.get("HRI_PROFILE")
    if env:
        return env
    return "pi5" if is_raspberry_pi() else "laptop"


def system_summary() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "board": board_model() or "-",
        "cpu_count": os.cpu_count(),
    }
