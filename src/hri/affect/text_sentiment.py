"""Tiny Vietnamese lexicon sentiment, used only for text-vs-face incongruence.
TODO: replace with a small local model (e.g. PhoBERT-based sentiment exported to ONNX)."""
from __future__ import annotations

POS = ["vui", "hạnh phúc", "tuyệt", "tốt", "thích", "yêu", "ổn", "khỏe", "hào hứng", "cảm ơn", "may mắn"]
NEG = ["buồn", "chán", "mệt", "tệ", "ghét", "lo", "sợ", "giận", "khóc", "cô đơn", "áp lực", "thất vọng", "đau"]
NEGATORS = ["không", "chẳng", "chả", "đâu có"]


def text_valence(text: str) -> float | None:
    t = f" {text.lower()} "
    score, hits = 0.0, 0
    for words, sign in ((POS, 1.0), (NEG, -1.0)):
        for w in words:
            idx = t.find(f" {w}")
            if idx >= 0:
                before = t[max(0, idx - 12):idx]
                s = -sign if any(n in before for n in NEGATORS) else sign
                score += s
                hits += 1
    return None if hits == 0 else max(-1.0, min(1.0, score / hits))
