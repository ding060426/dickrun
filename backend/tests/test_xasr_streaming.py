import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.asr_engine import XASREngine


class _FakeStreamingAsr:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def get_final_result(self):
        return "实时识别结果"


class XASRStreamingSessionTests(unittest.TestCase):
    def test_start_session_lazily_creates_streaming_recognizer(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)

        recognizer = _FakeStreamingAsr()
        engine.model_available = True
        engine._create_asr = lambda: recognizer

        engine.start_session()

        self.assertIs(engine.asr, recognizer)
        self.assertTrue(engine._session_active)
        self.assertEqual(recognizer.reset_calls, 0)

    def test_finalize_utterance_returns_final_result_and_opens_next_stream(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)

        recognizer = _FakeStreamingAsr()
        engine.asr = recognizer
        engine._session_active = True

        result = engine.finalize_utterance(reset_stream=True)

        self.assertEqual(result.raw_text, "实时识别结果")
        self.assertTrue(result.is_final)
        self.assertEqual(recognizer.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
