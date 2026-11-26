"""Child side: face selection, baseline, groups, reactions, micro-loop, distress."""
from types import SimpleNamespace

import numpy as np
import pytest

from hri.affect.child_state import (
    CONFUSED,
    DISLIKE,
    DISLIKED,
    EMOTION_TO_GROUP,
    ENJOY,
    FOCUS,
    GROUPS,
    LIKED,
    UNCLEAR,
    ChildQueue,
    ContentLog,
    group_probs,
    verdict,
)
from hri.affect.risk import adult_crisis, child_distress
from hri.config import load_config
from hri.dialogue.micro_loop import decide, pace
from hri.vision.face_tracker import CentralFaceTracker
from hri.vision.fer import FERPLUS_CLASSES as L
from hri.vision.postproc import FERStream

CFG = load_config("pi5")


def face(x, y, s=100):
    return SimpleNamespace(box=(x - s / 2, y - s / 2, s, s))


# ---------------------------------------------------------------- central face
def test_ignores_larger_face_outside_zone():
    tr = CentralFaceTracker()
    child, parent = face(330, 240, 80), face(80, 200, 220)
    assert tr.select([parent, child], 640, 480, 0.0) == (child, 1)
    assert tr.select([parent], 640, 480, 0.1) == (None, 1)


def test_follows_child_when_parent_leans_in_then_switches():
    tr = CentralFaceTracker({"switch_sec": 2})
    child, parent = face(300, 240), face(420, 200, 220)
    tr.select([child], 640, 480, 0.0)
    assert tr.select([parent, child], 640, 480, 0.1)[0] is child
    assert tr.select([parent], 640, 480, 0.2) == (None, 1)       # child hidden: do not read the parent's face
    assert tr.select([parent], 640, 480, 2.5) == (parent, 2)     # stayed: followed, but as another track


def test_look_away_keeps_track_long_absence_does_not():
    tr = CentralFaceTracker({"keep_sec": 10})
    c = face(320, 240)
    tr.select([c], 640, 480, 0.0)
    assert tr.select([], 640, 480, 3.0) == (None, 1)
    assert tr.select([face(340, 250)], 640, 480, 6.0)[1] == 1
    assert tr.select([c], 640, 480, 30.0)[1] == 2


# ---------------------------------------------------------------- baseline + stream
REST = np.array([.55, .05, .03, .25, .05, .03, .02, .02], np.float32)   # resting face that looks a bit sad


def feed(st, p, n, t=0.0, fps=8, **kw):
    f = None
    for i in range(n):
        f = st.update({"type": "fer", "t": t + i / fps, "probs": p, **kw})
    return f


def test_new_track_gets_its_own_baseline():
    st = FERStream(L, CFG.section("vision"))
    st.start_calibration()
    assert feed(st, REST, 40, track=1).calibrated
    assert not feed(st, REST, 1, t=6, track=2).calibrated
    assert feed(st, REST, 40, t=7, track=2).calibrated


# ---------------------------------------------------------------- groups and queue
def frame(t, label, p=0.8, cal=True, raw=None):
    probs = np.full(8, (1 - p) / 7)
    probs[L.index(label)] = p
    return SimpleNamespace(t=t, label=label, probs=probs, raw=probs if raw is None else raw, quality=0.8, calibrated=cal)


def queue(*parts, fps=8, cal=True):
    """parts: (label or None, seconds) -> queue, end time"""
    q, t = ChildQueue(L, CFG.section("child")), 0.0
    for label, sec in parts:
        for _ in range(int(sec * fps)):
            t += 1 / fps
            if label is None:
                q.add_absent(t)
            else:
                q.add(frame(t, label, cal=cal))
    return q, t


def test_every_class_has_one_group():
    assert set(EMOTION_TO_GROUP) == set(L) and set(EMOTION_TO_GROUP.values()) == set(GROUPS)
    assert group_probs(np.full(8, 1 / 8), L).tolist() == pytest.approx([1 / 8, 2 / 8, 2 / 8, 3 / 8])


def test_summary_dominance_trend_and_stability():
    q, t = queue(("neutral", 2), ("happiness", 4))
    s = q.summary(0, t)
    assert s["group"] == ENJOY and s["dominance"] == pytest.approx(0.67, abs=0.02)
    assert s["trend"] == ENJOY and 0 < s["switches_per_s"] < 0.5 and s["calibrated"]
    assert q.summary(t + 5, t + 6) == {"visible": False, "reason": "no_data", "n": 0}


def test_missing_face_means_away():
    q, t = queue(("neutral", 1), (None, 3))
    s = q.summary(t - 3, t)
    assert not s["visible"] and s["reason"] == "away" and s["group"] == DISLIKE
    assert q.summary(t - 1, t)["reason"] == "no_data"                    # a blink-long gap is not "away"


@pytest.mark.parametrize("parts, expected", [
    ([("happiness", 4)], LIKED), ([("surprise", 2), ("neutral", 2)], LIKED), ([("neutral", 4)], UNCLEAR),
    ([("disgust", 3), ("neutral", 1)], DISLIKED), ([(None, 4)], DISLIKED), ([("sadness", 4)], DISLIKED),
])
def test_reaction_verdict(parts, expected):
    q, t = queue(*parts)
    assert verdict(q.summary(0, t))[0] == expected


def test_uncalibrated_face_gives_no_verdict_and_no_pace_change():
    q, t = queue(("sadness", 4), cal=False)
    s = q.summary(0, t)
    assert verdict(s)[0] == UNCLEAR and pace(s, CFG.section("micro_loop")) == 1.0


def test_content_log_likes_and_dislikes():
    c = ContentLog()
    q1, t1 = queue(("happiness", 4))
    q2, t2 = queue(("anger", 4))
    c.add("con mèo", q1.summary(0, t1))
    c.add("con sư tử", q2.summary(0, t2))
    c.add("con cá", queue(("neutral", 4))[0].summary(0, 4))
    assert c.likes() == ["con mèo"] and c.dislikes() == ["con sư tử"]


# ---------------------------------------------------------------- robot reactions
@pytest.mark.parametrize("label, group, expr", [
    ("neutral", FOCUS, "listening"), ("happiness", ENJOY, "happy"), ("surprise", ENJOY, "curious"),
    ("sadness", CONFUSED, "empathy"), ("fear", CONFUSED, "concerned"), ("anger", DISLIKE, "calm"),
])
def test_micro_loop(label, group, expr):
    q, t = queue((label, 2))                            # dislike must last 1 s, so look at a 2-s stretch
    d = decide(q.summary(t - 1, t), CFG.section("micro_loop"))
    assert (d.group, d.expression) == (group, expr) and 0.3 <= d.intensity <= 0.8


def test_micro_loop_away_and_unclear():
    m = CFG.section("micro_loop")
    q, t = queue((None, 3))
    assert decide(q.summary(0, t), m).expression == "curious"
    q, t = queue(("neutral", 1), ("anger", 1.5), ("happiness", 1))
    assert decide(q.summary(0, t), m) is None


def test_pace_slows_down_when_confused():
    q, t = queue(("sadness", 3))
    assert pace(q.summary(0, t), CFG.section("micro_loop")) == 0.85


def test_distress_needs_long_calibrated_visible_upset():
    d = CFG.section("child.distress")
    for parts, cal, expected in [([("sadness", 5)], True, True), ([("sadness", 3), ("anger", 2)], True, True),
                                 ([("sadness", 2)], True, False), ([("sadness", 5)], False, False),
                                 ([(None, 5)], True, False), ([("sadness", 5), ("neutral", 2)], True, False)]:
        q, t = queue(*parts, cal=cal)
        assert child_distress(q, t, d) is expected, parts
    q, t = queue(("sadness", 5))
    assert not child_distress(q, t + 10, d)                              # stale data


def test_adult_crisis_keyword():
    assert adult_crisis("mình muốn chết", CFG.section("safety"))
    assert not adult_crisis("kể chuyện con mèo", CFG.section("safety"))


# ---------------------------------------------------------------- dislike needs support from the raw model
def raw(**kw):
    p = np.array([kw.get(c, 0.0) for c in L])
    p[0] += 1 - p.sum()
    return p


def guarded(label, r, n=24, **cfg):
    q = ChildQueue(L, {**CFG.section("child"), **cfg})
    got = [q.add(frame(i / 8, label, raw=r)) for i in range(n)]
    return got[-1], q.summary(0, 10)


@pytest.mark.parametrize("label, r, expected", [
    ("disgust", raw(disgust=0.08), FOCUS),                     # boosted noise
    ("disgust", raw(disgust=0.20), FOCUS),
    ("disgust", raw(disgust=0.30), DISLIKE),
    ("disgust", raw(disgust=0.12, anger=0.08, contempt=0.06), DISLIKE),   # together
    ("anger", raw(anger=0.5), DISLIKE), ("contempt", raw(contempt=0.02), FOCUS),
    ("sadness", raw(sadness=0.03), CONFUSED),                  # sadness is never touched
    ("happiness", raw(happiness=0.05), ENJOY),
])
def test_dislike_guard(label, r, expected):
    g, s = guarded(label, r)
    assert g == expected == s["group"]


def test_rejected_dislike_has_no_side_effects():
    _, s = guarded("disgust", raw(disgust=0.08))
    assert s["mean_probs"][DISLIKE] == 0 and s["mean_probs"][FOCUS] > 0.7 and s["label"] == "neutral"
    assert verdict(s)[0] == UNCLEAR and pace(s, CFG.section("micro_loop")) == 1.0
    assert decide(s, CFG.section("micro_loop")).expression == "listening"
    q = ChildQueue(L, CFG.section("child"))
    for i in range(60):
        q.add(frame(i / 8, "disgust", raw=raw(disgust=0.08)))
    assert not child_distress(q, 59 / 8, CFG.section("child.distress"))


def test_dislike_guard_can_be_switched_off():
    assert guarded("disgust", raw(disgust=0.02), dislike_min_raw=0, dislike_hold_sec=0)[0] == DISLIKE


def test_dislike_must_last():
    q = ChildQueue(L, CFG.section("child"))
    seq = [("anger", 0.6)] * 6 + [("neutral", 0.0)] * 4 + [("anger", 0.6)] * 12     # 0.75 s frown, then 1.5 s
    got = [q.add(frame(i / 8, lab, raw=raw(anger=a))) for i, (lab, a) in enumerate(seq)]
    assert DISLIKE not in got[:18] and got[18:] == [DISLIKE] * 4                       # confirmed after 1 s
    groups = [e.group for e in q.window(0, 99)]
    assert groups[:10] == [FOCUS] * 10 and groups[10:] == [DISLIKE] * 12                # the whole run, not the frown
    assert q.summary(10 / 8, 21 / 8)["mean_probs"][DISLIKE] > 0.5
    assert decide(q.summary(20 / 8 - 1, 21 / 8), CFG.section("micro_loop")).expression == "calm"
    q.add_absent(22 / 8)                                                               # a lost frame does not reset
    assert q.add(frame(23 / 8, "anger", raw=raw(anger=0.6))) == DISLIKE
    q.reset()
    assert q.add(frame(30, "anger", raw=raw(anger=0.6))) == FOCUS
