import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from audio_buffer import AudioBuffer, load_audio_buffer


class AudioBufferTests(unittest.TestCase):
    def test_stereo_8k_audio_is_loaded_once_as_canonical_16k_mono(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "stereo.wav"
            time = np.arange(8000, dtype=np.float32) / 8000
            stereo = np.column_stack(
                [
                    0.25 * np.sin(2 * np.pi * 220 * time),
                    0.25 * np.sin(2 * np.pi * 440 * time),
                ]
            )
            sf.write(path, stereo, 8000)

            audio = load_audio_buffer(path)

        self.assertEqual(audio.sample_rate, 16000)
        self.assertEqual(audio.samples.dtype, np.float32)
        self.assertEqual(audio.samples.ndim, 1)
        self.assertAlmostEqual(audio.duration, 1.0, places=3)
        self.assertLessEqual(float(np.max(np.abs(audio.samples))), 1.0)

    def test_slice_is_clamped_to_the_canonical_timeline(self):
        audio = AudioBuffer(np.arange(16000, dtype=np.float32) / 16000)
        sliced = audio.slice(-2.0, 0.25)
        self.assertEqual(len(sliced), 4000)


if __name__ == "__main__":
    unittest.main()
