import threading
import time
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class RuntimeReloadTests(unittest.TestCase):
    def test_saves_during_reload_are_coalesced_into_one_followup_load(self):
        original_loader = main._load_xasr_engine
        first_started = threading.Event()
        release_first = threading.Event()
        calls = []

        def fake_loader():
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                first_started.set()
                self.assertTrue(release_first.wait(timeout=2))

        try:
            with main._xasr_reload_lock:
                main._xasr_reload_pending = False
                main._xasr_reload_worker_active = False
            main._load_xasr_engine = fake_loader

            main._schedule_xasr_reload()
            self.assertTrue(first_started.wait(timeout=2))
            main._schedule_xasr_reload()
            main._schedule_xasr_reload()
            release_first.set()

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with main._xasr_reload_lock:
                    active = main._xasr_reload_worker_active
                if not active:
                    break
                time.sleep(0.01)

            self.assertFalse(active)
            self.assertEqual(calls, [1, 2])
        finally:
            release_first.set()
            main._load_xasr_engine = original_loader
            with main._xasr_reload_lock:
                main._xasr_reload_pending = False
                main._xasr_reload_worker_active = False


if __name__ == "__main__":
    unittest.main()
