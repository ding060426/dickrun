from modules.hotword_processor import HotwordProcessor


def test_ascii_alias_rewrites_to_canonical_hotword():
    processor = HotwordProcessor(["BERT", "A/B Test", "Q3"])
    text, corrections = processor.rewrite("we fine tuned bat in q 3 for a b test")
    assert "BERT" in text
    assert "Q3" in text
    assert "A/B Test" in text
    assert any(item["corrected"] == "BERT" for item in corrections)
    assert any(item["method"] == "canonical_alias" for item in corrections)


def test_matched_terms_returns_known_hotwords():
    processor = HotwordProcessor(["BERT", "Transformer", "转化率"])
    assert processor.matched_terms("BERT improves 转化率") == ["BERT", "转化率"]
