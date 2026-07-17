import sys
import tempfile
import time
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_runtime import ModelRuntimeService


class _Store:
    def __init__(self, payload):
        self.payload = payload

    def load(self):
        return self.payload


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Pool:
    def __init__(self, model_dir, base_options=None):
        self.model_dir = Path(model_dir)
        self.base_options = base_options or {}
        self.live_engine = object()
        self.final_engine = object()
        self.reload_calls = 0
        self.hotwords = None
        self.closed = False

    def reload(self, recognition, hotwords):
        self.reload_calls += 1
        return {
            "live": {"chunk_ms": 160},
            "final": {"chunk_ms": 160},
            "shared_runtime": True,
            "file_vad_provider": "silero",
            "inference_threads": self.base_options.get("num_threads", 1),
        }

    def status(self):
        return {"reload_calls": self.reload_calls}

    def create_live_session(self):
        return "live-session"

    def create_final_session(self):
        return "final-session"

    def configure_hotwords(self, settings):
        self.hotwords = settings

    def close(self):
        self.closed = True


class ModelRuntimeServiceTests(unittest.TestCase):
    def test_load_publishes_pool_and_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            service = ModelRuntimeService(
                has_xasr=True,
                model_dir=root,
                runtime_config_store=_Store({"recognition": {}}),
                hotword_config_store=_Store({"enabled": True, "words": []}),
                logger=_Logger(),
                base_options={"num_threads": 12},
                pool_factory=_Pool,
            )
            status = service.load()

        self.assertEqual(status["inference_threads"], 12)
        self.assertEqual(service.create_live_session(), "live-session")
        self.assertEqual(service.create_final_session(), "final-session")

    def test_schedule_reload_coalesces_to_worker(self):
        with tempfile.TemporaryDirectory() as root:
            service = ModelRuntimeService(
                has_xasr=True,
                model_dir=root,
                runtime_config_store=_Store({"recognition": {}}),
                hotword_config_store=_Store({"enabled": True, "words": []}),
                logger=_Logger(),
                pool_factory=_Pool,
            )
            service.schedule_reload()
            deadline = time.time() + 2
            while time.time() < deadline and not service.pool:
                time.sleep(0.01)

        self.assertIsNotNone(service.pool)
        self.assertGreaterEqual(service.status()["reload_calls"], 1)

    def test_configure_hotwords_and_close(self):
        with tempfile.TemporaryDirectory() as root:
            service = ModelRuntimeService(
                has_xasr=True,
                model_dir=root,
                runtime_config_store=_Store({"recognition": {}}),
                hotword_config_store=_Store({"enabled": True, "words": []}),
                logger=_Logger(),
                pool_factory=_Pool,
            )
            service.load()
            pool = service.pool
            service.configure_hotwords({"words": [{"text": "会悟"}]})
            service.close()

        self.assertEqual(pool.hotwords, {"words": [{"text": "会悟"}]})
        self.assertTrue(pool.closed)
        self.assertIsNone(service.pool)


if __name__ == "__main__":
    unittest.main()
