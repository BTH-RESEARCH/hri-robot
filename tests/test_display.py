from hri.ui.display import choose_video_driver, display_help


def test_pi_os_lite_uses_kmsdrm():
    assert choose_video_driver({"sdl_videodriver": "auto"}, environ={}, platform="linux") == "kmsdrm"


def test_desktop_sessions_keep_sdl_default():
    assert choose_video_driver({"sdl_videodriver": "auto"}, environ={"DISPLAY": ":0"}, platform="linux") is None
    assert choose_video_driver({}, environ={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux") is None
    assert choose_video_driver({"sdl_videodriver": "auto"}, environ={}, platform="win32") is None


def test_explicit_settings_win():
    assert choose_video_driver({"sdl_videodriver": "auto"}, environ={"SDL_VIDEODRIVER": "x11"},
                               platform="linux") == "x11"
    assert choose_video_driver({"sdl_videodriver": "wayland"}, environ={}, platform="linux") == "wayland"
    assert choose_video_driver({"sdl_videodriver": None}, environ={}, platform="linux") == "kmsdrm"


def test_help_mentions_the_fixes(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "kmsdrm")
    msg = display_help("kmsdrm not available")
    assert "python3-pygame" in msg and "video,render,input" in msg and "cage" in msg
