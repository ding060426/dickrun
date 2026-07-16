import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main


class _Pool:
    def __init__(self, provider="qwen3"):
        self.provider = provider

    def status(self):
        return {"effective_provider": self.provider}

    def create_final_session(self):
        return "qwen-final" if self.provider == "qwen3" else "xasr-final"

    def create_live_session(self):
        return "xasr-live"


class _Pipeline:
    def __init__(self, fail_primary=False):
        self.fail_primary = fail_primary
        self.calls = []

    def process_file(self, path, engine, *, enable_diarization):
        self.calls.append((path, engine, enable_diarization))
        if self.fail_primary and engine == "qwen-final":
            raise RuntimeError("CUDA out of memory")
        return {"engine": engine}


class Qwen3LiveFallbackTests(unittest.TestCase):
    def test_qwen_failure_retries_canonical_recording_with_xasr(self):
        pipeline = _Pipeline(fail_primary=True)

        run, metadata = main._process_canonical_recording("mic.wav", _Pool(), pipeline)

        self.assertEqual(run, {"engine": "xasr-live"})
        self.assertEqual([call[1] for call in pipeline.calls], ["qwen-final", "xasr-live"])
        self.assertEqual(metadata["provider"], "xasr")
        self.assertTrue(metadata["fallback"])
        self.assertIn("CUDA out of memory", metadata["primary_error"])

    def test_successful_qwen_run_does_not_fallback(self):
        pipeline = _Pipeline()

        run, metadata = main._process_canonical_recording("mic.wav", _Pool(), pipeline)

        self.assertEqual(run, {"engine": "qwen-final"})
        self.assertEqual(metadata, {"provider": "qwen3", "fallback": False, "primary_error": ""})


if __name__ == "__main__":
    unittest.main()
