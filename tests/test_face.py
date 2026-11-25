import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from hri.config import load_config
from hri.dialogue.micro_loop import AWAY, BY_LABEL, REACTION
from hri.ui.expressions import DESIGN, NEUTRAL, PRESETS, FaceController, target_params
from hri.ui.face_renderer import FaceRenderer


def renderer(**kw):
    c = FaceController()
    return FaceRenderer({"width": 400, "height": 240, "idle_motion": False, **kw}, c), c


def pixels(r):
    return r.pg.surfarray.array3d(r.screen).astype(int).swapaxes(0, 1)


def lit(r, p):
    r.draw(p)
    return pixels(r)[..., 1] > 150


def test_everything_the_app_can_show_exists():
    allowed = load_config("pi5").get("expression.allowed")
    used = {*allowed, *REACTION.values(), *BY_LABEL.values(), AWAY, "listening", "thinking", "sleep"}
    assert used <= set(PRESETS)
    assert not {"angry", "suspicious"} & set(PRESETS)


def test_intensity_is_readable_and_bounded():
    assert target_params("happy", 0) == NEUTRAL and target_params("nope", 1) == NEUTRAL
    assert target_params("happy", 0.8).bot == pytest.approx(0.40)
    assert target_params("happy", 5).bot == pytest.approx(0.40)
    assert 0.2 < target_params("happy", 0.4).bot < 0.4


def test_shapes_follow_the_design():
    r, _ = renderer(glow=0.0, scanlines=False)
    n = lit(r, DESIGN["neutral"])
    ys, xs = np.where(n)
    assert (xs.min() + xs.max()) / 2 == pytest.approx(200, abs=2) and (ys.min() + ys.max()) / 2 == pytest.approx(120, abs=2)
    assert xs.max() - xs.min() == pytest.approx((2 * 162 + 100) / 2, abs=4) and ys.max() - ys.min() == pytest.approx(106, abs=4)
    assert not n[:, 195:205].any()                                          # gap between the eyes
    h = lit(r, DESIGN["happy"])
    assert not h[145:].any() and h[80:110].any()                            # lower lid up
    s = lit(r, DESIGN["sad"])
    left = np.where(s[:, 100:180].any(0))[0]
    top = [np.where(s[:, 100 + c])[0].min() for c in (left[4], left[-5])]
    assert top[0] > top[1]                                                  # inner end of the lid is higher
    assert lit(r, DESIGN["surprised"]).sum() > 1.2 * n.sum() > 0
    assert lit(r, DESIGN["sleep"]).sum() < 0.05 * n.sum()


def test_curious_is_the_neutral_face_a_little_bigger():
    c, n = target_params("curious", 0.8), NEUTRAL
    assert c.scale > n.scale and {**vars(c), "scale": 1.0} == vars(n)


@pytest.mark.parametrize("size", [(1024, 600), (800, 480)])
@pytest.mark.parametrize("name", list(PRESETS))
def test_eyes_stay_on_the_lcd_at_the_extremes(name, size):
    """Furthest gaze, loudest voice, whole face tilted: no eye is cut by the edge of the screen."""
    r, _ = renderer(width=size[0], height=size[1], glow=0.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = target_params(name, 1.0)
            gaze = 0 if name == "thinking" else 1
            p.dx += sx * gaze * 0.75 * 58
            p.dy += sy * (gaze * 0.45 * 34 + 7)
            p.h *= 1.10
            p.scale *= 1.03
            on = lit(r, p) if p.dim == 1 else np.zeros(size[::-1], bool)
            assert not (on[0].any() or on[-1].any() or on[:, 0].any() or on[:, -1].any())


def test_halo_is_centred_on_the_eye():
    r, _ = renderer(glow=0.55, scanlines=False)
    eye = r._eye(1, DESIGN["neutral"], 0, 1, 0, 0)
    h = r.pg.surfarray.array3d(eye[1])[..., 1].astype(float)
    cx = (h.sum(1) * np.arange(h.shape[0])).sum() / h.sum() - (h.shape[0] - 1) / 2
    cy = (h.sum(0) * np.arange(h.shape[1])).sum() / h.sum() - (h.shape[1] - 1) / 2
    assert abs(cx) < 2 and abs(cy) < 3                                      # gradient makes the top a bit brighter


@pytest.mark.parametrize("name", list(PRESETS))
def test_animation_runs_for_every_expression(name):
    r, c = renderer(idle_motion=True)
    c.set_expression(name, 0.8)
    for i in range(90):
        c.set_level(0.5 + 0.5 * np.sin(i))
        r.step(1 / 30, now=i / 30)
    assert len(r._cache) <= 96


def test_voice_moves_the_eyes_and_blink_closes_them():
    r, c = renderer(glow=0.0, scanlines=False)
    for i in range(60):
        r.step(1 / 30, now=i / 30)
    quiet = (pixels(r)[..., 1] > 150).sum()
    c.set_level(1.0)
    for i in range(60, 90):
        r.step(1 / 30, now=i / 30)
    assert (pixels(r)[..., 1] > 150).sum() > 1.05 * quiet
    r.draw(DESIGN["neutral"], blink=1.0)
    assert (pixels(r)[..., 1] > 150).sum() < 0.1 * quiet


def test_tiny_panel_does_not_crash():
    r, c = renderer(width=120, height=40)
    for n in PRESETS:
        c.set_expression(n, 1.0)
        r.step(0.05)
