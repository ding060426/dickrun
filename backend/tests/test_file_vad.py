import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.file_vad import SileroFileVad


class _FakeDetector:
    def __init__(self):
        self._segments = []

    def accept_waveform(self, samples):
        pass

    def flush(self):
        self._segments.extend([
            SimpleNamespace(start=1600, samples=np.zeros(3200, dtype=np.float32)),
            SimpleNamespace(start=5600, samples=np.zeros(1600, dtype=np.float32)),
        ])

    def empty(self):
        return not self._segments

    @property
    def front(self):
        return self._segments[0]

    def pop(self):
        self._segments.pop(0)


class SileroFileVadTests(unittest.TestCase):
    def test_returns_padded_merged_speech_regions_in_seconds(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "silero_vad.onnx"
            model.write_bytes(b"test")
            vad = SileroFileVad(
                model,
                pre_padding_ms=100,
                post_padding_ms=200,
                detector_factory=lambda _: _FakeDetector(),
            )
            audio = np.zeros(16000, dtype=np.float32)

            regions = vad.detect(audio, 16000)

        self.assertEqual(regions, [(0.0, 0.65)])
        self.assertEqual(vad.provider_name, "sherpa-silero-file-vad")


if __name__ == "__main__":
    unittest.main()
