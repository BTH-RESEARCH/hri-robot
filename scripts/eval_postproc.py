"""Compare post-processing variants on the FER pack's test images (no webcam needed).

Simulated person = one neutral image (resting face, 4 s) then one expression image (2 s), with correlated noise
on the logits. Images come from DIFFERENT people (FERPlus has no identities), so this is a sanity check of the
filter, not a user study, and does not replace measuring on real video.

    python scripts/eval_postproc.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path

import numpy as np

from hri.config import load_config
from hri.vision.fer import FERPLUS_CLASSES as L
from hri.vision.fer import softmax
from hri.vision.postproc import FERStream

G = {"neutral": 0, "happiness": 1, "surprise": 1, "sadness": 2, "fear": 2, "anger": 3, "disgust": 3, "contempt": 3}


def load(cfg):
    from hri.models import ModelRegistry
    d = ModelRegistry(cfg).require(cfg.get("vision.fer.model")) / "testset"
    z = np.load(d / "ref_logits.npz")["fp32"]
    with open(Path(d) / "labels.csv", encoding="utf-8") as f:
        y = np.array([int(r["label"]) for r in csv.DictReader(f)])
    return z, y


def split(y, half, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for k in range(8):
        i = rng.permutation(np.where(y == k)[0])
        idx += list(i[: len(i) // 2] if half == "A" else i[len(i) // 2:])
    return np.array(idx)


def noisy(z, n, rng, sd):
    e = np.zeros(8)
    out = []
    for _ in range(n):
        e = 0.7 * e + rng.normal(0, sd, 8)
        out.append(softmax(z + e))
    return out


class EMA:
    def __init__(self, a):
        self.a, self.e = a, None

    def step(self, t, p):
        self.e = p if self.e is None else self.a * p + (1 - self.a) * self.e
        return L[int(self.e.argmax())]


class Chain:
    def __init__(self, vcfg, fps):
        self.s = FERStream(L, {**vcfg, "target_fps": fps})
        self.s.start_calibration()

    def step(self, t, p):
        return self.s.update({"type": "fer", "t": t, "probs": p}).label


def evaluate(make, z, y, half, fps=8, sd=0.3, limit=None):
    """make(fps) -> object with step(t, probs) -> label"""
    idx = split(y, half)
    neu, expr = np.random.default_rng(2).permutation(idx[y[idx] == 0]), idx[y[idx] != 0]
    if limit:
        neu, expr = neu[:limit], expr[::max(1, len(expr) // (2 * limit))]   # keeps every class
    rng = np.random.default_rng(1)
    rest_ok, rest_sw, hit8, hit4, delay = [], [], {}, {}, []
    for j, i in enumerate(expr):
        m = make(fps)
        n0 = neu[j % len(neu)]
        t, lab = 0.0, []
        for p in noisy(z[n0], 5 * fps, rng, sd):
            lab.append(m.step(t, p))
            t += 1 / fps
        tail = lab[-2 * fps:]
        rest_ok.append(np.mean([x == "neutral" for x in tail]))
        rest_sw.append(sum(a != b for a, b in itertools.pairwise(tail)) / 2)
        lab = []
        for p in noisy(z[i], 2 * fps, rng, sd):
            lab.append(m.step(t, p))
            t += 1 / fps
        k = int(y[i])
        hit8.setdefault(k, []).append(np.mean([x == L[k] for x in lab[fps:]]))
        ok = [G[x] == G[L[k]] for x in lab]
        hit4.setdefault(G[L[k]], []).append(np.mean(ok[fps:]))
        # onset delay only where it means something: the raw model sees the class, and rest was really neutral
        if z[i].argmax() == k and tail[-1] == "neutral" and any(ok):
            delay.append(ok.index(True) / fps)
    r = {"rest_neutral": np.mean(rest_ok), "rest_flips_per_s": np.mean(rest_sw),
         "recall8": np.mean([np.mean(v) for v in hit8.values()]),
         "recall4": np.mean([np.mean(rest_ok), *[np.mean(v) for v in hit4.values()]]),
         "groups": [round(float(np.mean(hit4[g])), 2) for g in sorted(hit4)],
         "delay_s": float(np.median(delay)) if delay else float("nan")}
    return r


def show(name, r):
    print(f"{name:26s} rest=neutral {100 * r['rest_neutral']:5.1f}%  flips/s {r['rest_flips_per_s']:.2f}  "
          f"recall(4 groups) {100 * r['recall4']:5.1f}%  [joy,sad,dislike] {r['groups']}  "
          f"recall(7 expr) {100 * r['recall8']:5.1f}%  onset {r['delay_s']:.2f}s")


def variants(cfg):
    v = cfg.section("vision")
    raw = copy.deepcopy(v)
    for k in ("sensitivity", "baseline", "hysteresis"):
        raw[k]["enabled"] = False
    out = {"raw argmax": lambda fps: Chain(raw, fps), "EMA 0.30": lambda fps: EMA(0.30),
           "EMA 0.45": lambda fps: EMA(0.45)}
    for off in ("sensitivity", "baseline", "hysteresis"):
        c = copy.deepcopy(v)
        c[off]["enabled"] = False
        out[f"V3b without {off}"] = lambda fps, c=c: Chain(c, fps)
    c = copy.deepcopy(v)
    c["hysteresis"]["transition_dynamics"] = False
    out["V3b without bio rules (V3a)"] = lambda fps, c=c: Chain(c, fps)
    out["V3b (configured)"] = lambda fps: Chain(v, fps)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config("pi5")
    z, y = load(cfg)
    print(f"== half B, {a.fps} FPS")
    for name, mk in variants(cfg).items():
        show(name, evaluate(mk, z, y, "B", a.fps))
