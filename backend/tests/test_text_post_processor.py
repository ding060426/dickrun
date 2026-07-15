import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.text_post_processor import (
    inverse_text_normalize,
    process_asr_text,
    process_asr_text_with_details,
    restore_punctuation,
)


class _TrackingPunctuationModel:
    is_available = True

    def __init__(self):
        self.calls = 0

    def add_punctuation(self, text):
        self.calls += 1
        return text + "。"


class TextPostProcessorTests(unittest.TestCase):
    def test_detailed_pipeline_reports_fillers_and_repetitions(self):
        text, details = process_asr_text_with_details("嗯，就是说好好好我们开始开会啊")

        self.assertTrue(text)
        self.assertIn("就是说", details["fillers_removed"])
        self.assertIn("好好好", details["repetitions_merged"])
        self.assertEqual(details["original_text"], "嗯，就是说好好好我们开始开会啊")

    def test_macbert_is_opt_in_and_reports_corrections(self):
        class _Corrector:
            def correct_batch(self, texts):
                return [{
                    "target": "今天针对产品开会。",
                    "errors": [["真对", "针对", 2]],
                }]

        with (
            patch.dict("os.environ", {"DITING_ENABLE_MACBERT": "1"}),
            patch(
                "modules.text_post_processor._get_macbert_corrector",
                return_value=_Corrector(),
            ),
        ):
            text, details = process_asr_text_with_details("今天真对产品开会")

        self.assertEqual(text, "今天针对产品开会。")
        self.assertEqual(details["corrections"][0]["corrected"], "针对")

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

    def test_inverse_text_normalization_handles_common_dictation_numbers(self):
        source = "二零二六年百分之二十五下午三点半端口八零八零预算五千八百元"

        result = inverse_text_normalize(source)

        self.assertEqual(result, "2026年25%下午3:30端口8080预算5800元")

    def test_filler_cleanup_preserves_meaningful_words_and_short_repetition(self):
        result = process_asr_text(
            "嗯金额额外看看那个那个我们开始",
            enable_punctuation=False,
            enable_force_split=False,
        )

        self.assertEqual(result, "金额额外看看那个我们开始")

    def test_detailed_pipeline_reports_number_normalization(self):
        text, details = process_asr_text_with_details("预算一百二十三元")

        self.assertIn("123元", text.replace(" ", ""))
        self.assertEqual(details["itn_original"], "预算一百二十三元")
        self.assertIn("123元", details["itn_normalized"])


if __name__ == "__main__":
    unittest.main()
