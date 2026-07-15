import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.hotword_config import HotwordConfigStore


class HotwordConfigStoreTests(unittest.TestCase):
    def test_seeds_defaults_with_safe_language_specific_scores(self):
        with tempfile.TemporaryDirectory() as root:
            store = HotwordConfigStore(Path(root) / "hotwords.json", ["BERT", "贾扬清"])

            settings = store.load()

            self.assertEqual(settings["words"][0], {
                "text": "BERT", "score": 2.5, "enabled": True,
            })
            self.assertEqual(settings["words"][1]["score"], 5.0)
            self.assertTrue(settings["fuzzy_pinyin_enabled"])

    def test_persists_normalized_entries_and_deduplicates_ascii_case(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "hotwords.json"
            store = HotwordConfigStore(path, [])

            saved = store.save({
                "enabled": True,
                "fuzzy_pinyin_enabled": False,
                "default_score": 6,
                "words": [
                    {"text": " OpenAI ", "score": 3.5, "enabled": True},
                    {"text": "openai", "score": 8, "enabled": True},
                    {"text": "贾扬清", "score": 7, "enabled": False},
                    {"text": "# ignored", "score": 5, "enabled": True},
                ],
            })

            self.assertEqual(len(saved["words"]), 2)
            self.assertEqual(saved["words"][0]["text"], "OpenAI")
            self.assertEqual(saved["words"][0]["score"], 3.5)
            self.assertFalse(saved["fuzzy_pinyin_enabled"])
            self.assertEqual(HotwordConfigStore(path, []).load(), saved)

    def test_engine_inputs_include_only_enabled_words_and_scores(self):
        with tempfile.TemporaryDirectory() as root:
            store = HotwordConfigStore(Path(root) / "hotwords.json", [])
            settings = store.save({
                "enabled": True,
                "fuzzy_pinyin_enabled": True,
                "default_score": 5,
                "words": [
                    {"text": "BERT", "score": 2.5, "enabled": True},
                    {"text": "关闭词", "score": 8, "enabled": False},
                ],
            })

            words, scores = store.engine_inputs(settings)

            self.assertEqual(words, ["BERT"])
            self.assertEqual(scores, {"BERT": 2.5})


if __name__ == "__main__":
    unittest.main()
