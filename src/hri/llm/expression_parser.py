"""Streaming parser for LLM output like:
    [EMO:happy|0.8] Chào bạn! [EMO:curious|0.5] Hôm nay bạn thế nào?
Feed token deltas; it emits Segment(emotion, intensity, text) as soon as a sentence is complete,
so TTS can speak the first sentence while the LLM is still generating.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TAG_RE = re.compile(r"\[EMO:\s*([a-zA-Z_]+)\s*(?:\|\s*(-?[0-9]*\.?[0-9]+))?\s*\]")
SENT_END_RE = re.compile(r"([.!?…]+)(\s+|$)")
# anything TTS should not read: markdown symbols, emoji, brackets
CLEAN_RE = re.compile(r"[*_#`>~\[\]]|[\U0001F000-\U0001FAFF☀-➿]")
MAX_TAG_LEN = 40


@dataclass
class Segment:
    emotion: str
    intensity: float
    text: str


class ExpressionStreamParser:
    def __init__(self, allowed: list[str], default: str = "neutral", soft_split_chars: int = 80):
        self.allowed = set(allowed)
        self.default = default
        self.soft_split = soft_split_chars
        self.emotion, self.intensity = default, 0.5
        self._raw = ""    # unparsed input
        self._text = ""   # plain text under the current emotion, not yet emitted

    def feed(self, delta: str) -> list[Segment]:
        self._raw += delta
        return self._drain(final=False)

    def flush(self) -> list[Segment]:
        return self._drain(final=True)

    # ------------------------------------------------------------------
    def _normalize(self, name: str, value: str | None) -> tuple[str, float]:
        name = name.lower()
        emo = name if name in self.allowed else self.default
        try:
            inten = float(value) if value else 0.6
        except ValueError:
            inten = 0.6
        return emo, max(0.0, min(1.0, inten))

    def _drain(self, final: bool) -> list[Segment]:
        out: list[Segment] = []
        while self._raw:
            lb = self._raw.find("[")
            if lb == -1:
                self._text += self._raw
                self._raw = ""
                break
            self._text += self._raw[:lb]
            self._raw = self._raw[lb:]
            m = TAG_RE.match(self._raw)
            if m:
                out += self._take(force=True)      # emotion changes -> flush text of the old one
                self.emotion, self.intensity = self._normalize(m.group(1), m.group(2))
                self._raw = self._raw[m.end():]
                continue
            if not final and len(self._raw) < MAX_TAG_LEN and "]" not in self._raw:
                break                               # tag may still be streaming in
            self._text += "["                       # not a tag: literal '[' (cleaned later)
            self._raw = self._raw[1:]
        out += self._take(force=final)
        return out

    def _take(self, force: bool) -> list[Segment]:
        segs: list[Segment] = []
        pos = 0
        text = self._text
        for m in SENT_END_RE.finditer(text):
            segs.append(self._seg(text[pos:m.end(1)]))
            pos = m.end()
        tail = text[pos:]
        if not force and len(tail) > self.soft_split and "," in tail:
            cut = tail.rfind(",") + 1               # long clause: start TTS at a comma
            segs.append(self._seg(tail[:cut]))
            tail = tail[cut:]
        if force:
            segs.append(self._seg(tail))
            tail = ""
        self._text = tail
        return [s for s in segs if s.text]

    def _seg(self, text: str) -> Segment:
        clean = re.sub(r"\s+", " ", CLEAN_RE.sub("", text)).strip()
        return Segment(self.emotion, self.intensity, clean)


def parse_full(text: str, allowed: list[str], default: str = "neutral") -> list[Segment]:
    p = ExpressionStreamParser(allowed, default)
    return p.feed(text) + p.flush()
