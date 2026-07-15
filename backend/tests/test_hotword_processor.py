import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.hotword_processor import HotwordProcessor, fuzzy_pinyin_key


class HotwordProcessorTests(unittest.TestCase):
    def test_fuzzy_key_collapses_common_accent_confusions(self):
        self.assertEqual(fuzzy_pinyin_key("shi"), "si")
        self.assertEqual(fuzzy_pinyin_key("niu"), "liu")
        self.assertEqual(fuzzy_pinyin_key("jing"), "jin")
        self.assertEqual(fuzzy_pinyin_key("zhang"), "zan")

    def test_rewrites_multi_character_fuzzy_pinyin_longest_first(self):
        processor = HotwordProcessor(["世界", "世界银行"], fuzzy_pinyin_enabled=True)

        text, corrections = processor.rewrite("四界银航今天开会")

        self.assertEqual(text, "世界银行今天开会")
        self.assertEqual(corrections[0]["method"], "fuzzy_pinyin")
        self.assertEqual(corrections[0]["original"], "四界银航")

    def test_single_character_hotwords_never_drive_fuzzy_rewrites(self):
        processor = HotwordProcessor(["木"], fuzzy_pinyin_enabled=True)

        text, corrections = processor.rewrite("目标")

        self.assertEqual(text, "目标")
        self.assertEqual(corrections, [])

    def test_canonicalizes_ascii_spacing_case_and_known_aliases(self):
        processor = HotwordProcessor(["BERT", "OpenAI"], fuzzy_pinyin_enabled=False)

        text, corrections = processor.rewrite("我们使用 b e r t 和 open ai，也有人念成 bat")

        self.assertEqual(text, "我们使用 BERT 和 OpenAI，也有人念成 BERT")
        self.assertTrue(all(item["method"] == "canonical_alias" for item in corrections))


if __name__ == "__main__":
    unittest.main()
