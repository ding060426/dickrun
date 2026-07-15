import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from start import API_REVISION, is_compatible_backend
from backend.build_info import API_REVISION as BACKEND_API_REVISION


class StartupCompatibilityTests(unittest.TestCase):
    def test_launcher_and_backend_share_one_revision(self):
        self.assertEqual(API_REVISION, BACKEND_API_REVISION)

    def test_rejects_stale_diting_process_without_current_api_revision(self):
        self.assertFalse(is_compatible_backend({"service": "会悟 v2.0"}))

    def test_accepts_current_backend_revision(self):
        self.assertTrue(
            is_compatible_backend(
                {"service": "会悟 v2.0", "api_revision": API_REVISION}
            )
        )


if __name__ == "__main__":
    unittest.main()
