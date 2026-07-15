import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.runtime_config import RuntimeConfigStore


class RuntimeConfigStoreTests(unittest.TestCase):
    def test_defaults_use_local_960ms_models_and_silero_file_segmentation(self):
        with tempfile.TemporaryDirectory() as root:
            settings = RuntimeConfigStore(Path(root) / "settings.json").load()

        self.assertEqual(settings["recognition"]["live_asr_profile"], "meeting")
        self.assertEqual(settings["recognition"]["final_asr_profile"], "meeting")
        self.assertEqual(settings["recognition"]["file_vad_provider"], "silero")
        self.assertFalse(settings["microphone"]["vad_gating"])

    def test_save_clamps_latency_and_vad_values(self):
        with tempfile.TemporaryDirectory() as root:
            store = RuntimeConfigStore(Path(root) / "settings.json")
            settings = store.save({
                "recognition": {
                    "live_asr_profile": "quality",
                    "final_asr_profile": "invalid",
                    "file_vad_threshold": 8,
                },
                "microphone": {
                    "endpoint_grace_ms": 99999,
                    "pre_roll_ms": -1,
                    "vad_min_speech": 0,
                },
            })

            reloaded = store.load()

        self.assertEqual(settings, reloaded)
        self.assertEqual(settings["recognition"]["live_asr_profile"], "quality")
        self.assertEqual(settings["recognition"]["final_asr_profile"], "meeting")
        self.assertEqual(settings["recognition"]["file_vad_threshold"], 0.95)
        self.assertEqual(settings["microphone"]["endpoint_grace_ms"], 5000)
        self.assertEqual(settings["microphone"]["pre_roll_ms"], 0)
        self.assertEqual(settings["microphone"]["vad_min_speech"], 0.05)


if __name__ == "__main__":
    unittest.main()
