import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.download_models import profile_files


class ModelDownloadTests(unittest.TestCase):
    def test_meeting_profile_resolves_official_960ms_artifacts(self):
        files = profile_files("meeting")

        self.assertEqual([name for _, name in files], [
            "encoder-960ms.onnx",
            "decoder-960ms.onnx",
            "joiner-960ms.onnx",
            "tokens.txt",
        ])
        self.assertTrue(all("chunk-960ms-model" in remote for remote, _ in files))


if __name__ == "__main__":
    unittest.main()
