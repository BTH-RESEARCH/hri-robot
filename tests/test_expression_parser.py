from hri.llm.expression_parser import ExpressionStreamParser, parse_full

ALLOWED = ["neutral", "happy", "sad", "empathy", "curious"]


def test_full_parse():
    segs = parse_full("[EMO:happy|0.8] Chào bạn! [EMO:curious|0.5] Hôm nay bạn thế nào?", ALLOWED)
    assert [(s.emotion, s.intensity, s.text) for s in segs] == [
        ("happy", 0.8, "Chào bạn!"), ("curious", 0.5, "Hôm nay bạn thế nào?")]


def test_streaming_char_by_char_matches_full():
    text = "[EMO:empathy|0.7] Ôi, chắc bạn buồn lắm. Kể mình nghe nhé? [EMO:happy] Mình luôn ở đây."
    p = ExpressionStreamParser(ALLOWED)
    segs = []
    for ch in text:
        segs += p.feed(ch)
    segs += p.flush()
    assert [s.text for s in segs] == ["Ôi, chắc bạn buồn lắm.", "Kể mình nghe nhé?", "Mình luôn ở đây."]
    assert [s.emotion for s in segs] == ["empathy", "empathy", "happy"]
    assert segs[-1].intensity == 0.6  # default when intensity missing


def test_first_sentence_emitted_before_stream_ends():
    p = ExpressionStreamParser(ALLOWED)
    assert p.feed("[EMO:happy|0.9] Xin chào") == []
    out = p.feed(" bạn! Hôm")
    assert [s.text for s in out] == ["Xin chào bạn!"]


def test_unknown_label_falls_back_and_clamps():
    segs = parse_full("[EMO:angry|3] Hừm. [EMO:sad|-1] Thôi.", ALLOWED)
    assert (segs[0].emotion, segs[0].intensity) == ("neutral", 1.0)
    assert segs[1].emotion == "sad"


def test_no_tag_and_markdown_cleanup():
    segs = parse_full("**Chào** bạn 😊 [ghi chú] nhé", ALLOWED)
    assert segs[0].emotion == "neutral"
    assert segs[0].text == "Chào bạn ghi chú nhé"
