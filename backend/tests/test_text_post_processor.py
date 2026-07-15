import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.text_post_processor import process_asr_text, restore_punctuation


class _TrackingPunctuationModel:
    is_available = True

    def __init__(self):
        self.calls = 0

    def add_punctuation(self, text):
        self.calls += 1
        return text + "。"


class TextPostProcessorTests(unittest.TestCase):
    def test_skips_ml_restoration_when_asr_already_contains_punctuation(self):
        model = _TrackingPunctuationModel()
        with patch(
            "modules.text_post_processor._get_punct_restorer",
            return_value=model,
        ):
            result = restore_punctuation("大家好，今天开始开会。")

        self.assertEqual(result, "大家好，今天开始开会。")
        self.assertEqual(model.calls, 0)

    def test_collapses_adjacent_punctuation_using_strongest_boundary(self):
        result = process_asr_text(
            "大家好。，今天开始开会。。",
            enable_filler_filter=False,
            enable_punctuation=False,
            enable_force_split=False,
        )

        self.assertEqual(result, "大家好。今天开始开会。")


if __name__ == "__main__":
    unittest.main()
