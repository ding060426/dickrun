import sys
import tempfile
import unittest
import wave
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.recording import LiveRecording


class LiveRecordingTests(unittest.TestCase):
    def test_existing_recording_id_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            recording = LiveRecording(root, "duplicate")
            recording.append_pcm_s16le(b"\x00\x00" * 16)
            recording.finalize()

            with self.assertRaises(FileExistsError):
                LiveRecording(root, "duplicate")

    def test_finalize_atomically_publishes_a_valid_wav(self):
        with tempfile.TemporaryDirectory() as root:
            recording = LiveRecording(Path(root), "stream-1")
            recording.append_pcm_s16le((b"\x01\x00" * 640))
            recording.append_pcm_s16le((b"\x02\x00" * 320))

            result = recording.finalize()

            self.assertTrue(result.path.is_file())
            self.assertFalse(result.part_path.exists())
            self.assertEqual(result.received_samples, 960)
            self.assertEqual(result.duration_ms, 60)
            with wave.open(str(result.path), "rb") as source:
                self.assertEqual(source.getframerate(), 16000)
                self.assertEqual(source.getnchannels(), 1)
                self.assertEqual(source.getnframes(), 960)

    def test_abort_keeps_the_partial_recording_recoverable(self):
        with tempfile.TemporaryDirectory() as root:
            recording = LiveRecording(Path(root), "stream-2")
            recording.append_pcm_s16le((b"\x01\x00" * 640))

            recovery_path = recording.abort()

            self.assertEqual(recovery_path.suffix, ".part")
            self.assertTrue(recovery_path.is_file())


if __name__ == "__main__":
    unittest.main()
