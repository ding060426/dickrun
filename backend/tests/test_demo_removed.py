import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from fastapi.testclient import TestClient


class DemoRemovalTests(unittest.TestCase):
    def test_backend_does_not_expose_built_in_meeting_samples(self):
        paths = {route.path for route in main.app.routes}

        self.assertNotIn("/api/meeting/demo", paths)
        self.assertNotIn("/ws/meeting", paths)
        self.assertFalse(hasattr(main, "DEMO_MEETING"))

    def test_live_transcription_reports_unavailable_instead_of_fake_text(self):
        original_pool = main.xasr_pool
        original_engine = main.xasr_engine
        main.xasr_pool = None
        main.xasr_engine = None
        try:
            with TestClient(main.app).websocket_connect("/ws/live") as websocket:
                message = websocket.receive_json()
        finally:
            main.xasr_pool = original_pool
            main.xasr_engine = original_engine

        self.assertEqual(message["type"], "error")
        self.assertNotIn("demo", str(message).lower())


if __name__ == "__main__":
    unittest.main()
