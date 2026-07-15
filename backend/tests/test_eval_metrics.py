import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from eval_ali_integration import evaluate_against_textgrid


class EvaluationMetricTests(unittest.TestCase):
    def test_cer_uses_ground_truth_length_as_denominator(self):
        metrics = evaluate_against_textgrid(
            [{"text": "ab", "start_sec": 0, "end_sec": 1}],
            [{"text": "abcd", "start_sec": 0, "end_sec": 1}],
        )

        self.assertEqual(metrics["cer"], 0.5)
        self.assertEqual(metrics["gt_chars"], 4)
        self.assertEqual(metrics["asr_chars"], 2)


if __name__ == "__main__":
    unittest.main()
