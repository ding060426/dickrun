import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.engine_pool import AsrEnginePool


def _write_profile(root: Path, chunk_ms: int):
    (root / "tokens.txt").write_text("token 0\n", encoding="utf-8")
    for component in ("encoder", "decoder", "joiner"):
        (root / f"{component}-{chunk_ms}ms.onnx").write_bytes(b"model")


class _FakeEngine:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.asr_profile = kwargs["asr_profile"]
        self.chunk_ms = {
            "low-latency": 160,
            "balanced": 480,
            "meeting": 960,
            "quality": 1920,
        }[self.asr_profile]
        self.is_model_available = True
        self.warmed = False
        self.configured = []
        self.__class__.instances.append(self)

    def warmup(self):
        self.warmed = True
        return self

    def fork_session(self):
        return ("session", self.asr_profile)

    def configure_hotwords(self, *args, **kwargs):
        self.configured.append((args, kwargs))


class _FakeQwenEngine:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_model_available = True
        self.file_vad_provider = "qwen3-full-audio"
        self.logic_validator = None
        self.__class__.instances.append(self)

    @classmethod
    def availability(cls, model_path=""):
        return {"available": bool(model_path), "reason": "" if model_path else "model_not_configured"}

    def warmup(self):
        return self

    def fork_session(self):
        return ("qwen3-session", self.kwargs["model_path"])

    def configure_hotwords(self, *args, **kwargs):
        pass


class AsrEnginePoolTests(unittest.TestCase):
    def setUp(self):
        _FakeEngine.instances = []
        _FakeQwenEngine.instances = []

    def test_same_live_and_final_profile_share_one_loaded_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root)
            _write_profile(model_dir, 960)
            pool = AsrEnginePool(model_dir, engine_factory=_FakeEngine)

            status = pool.reload(
                {"live_asr_profile": "meeting", "final_asr_profile": "meeting"},
                {"enabled": True, "fuzzy_pinyin_enabled": True, "default_score": 5, "words": []},
            )

        self.assertEqual(len(_FakeEngine.instances), 1)
        self.assertIs(pool.live_engine, pool.final_engine)
        self.assertEqual(pool.create_live_session(), ("session", "meeting"))
        self.assertEqual(status["live"]["chunk_ms"], 960)
        self.assertTrue(status["shared_runtime"])

    def test_missing_requested_profile_falls_back_to_deployed_960ms(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root)
            _write_profile(model_dir, 960)
            pool = AsrEnginePool(model_dir, engine_factory=_FakeEngine)

            status = pool.reload(
                {"live_asr_profile": "quality", "final_asr_profile": "quality"},
                {"enabled": False, "fuzzy_pinyin_enabled": False, "default_score": 5, "words": []},
            )

        self.assertEqual(status["live"]["requested_profile"], "quality")
        self.assertEqual(status["live"]["effective_profile"], "meeting")
        self.assertTrue(status["live"]["fallback"])

    def test_qwen3_selection_keeps_live_xasr_and_switches_final_engine(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root)
            _write_profile(model_dir, 960)
            pool = AsrEnginePool(
                model_dir,
                engine_factory=_FakeEngine,
                qwen_engine_factory=_FakeQwenEngine,
            )

            status = pool.reload(
                {
                    "asr_provider": "qwen3",
                    "qwen3_model_path": "Qwen/Qwen3-ASR-0.6B",
                    "qwen3_device": "cuda:0",
                    "qwen3_dtype": "bfloat16",
                    "live_asr_profile": "meeting",
                },
                {"enabled": True, "fuzzy_pinyin_enabled": True, "default_score": 5, "words": []},
            )

        self.assertEqual(pool.create_live_session(), ("session", "meeting"))
        self.assertEqual(pool.create_final_session(), ("qwen3-session", "Qwen/Qwen3-ASR-0.6B"))
        self.assertEqual(len(_FakeEngine.instances), 1)
        self.assertEqual(status["selected_provider"], "qwen3")
        self.assertEqual(status["effective_provider"], "qwen3")
        self.assertEqual(status["live_provider"], "xasr")
        self.assertFalse(status["provider_fallback"])

    def test_unavailable_qwen3_falls_back_to_xasr(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root)
            _write_profile(model_dir, 960)
            pool = AsrEnginePool(
                model_dir,
                engine_factory=_FakeEngine,
                qwen_engine_factory=_FakeQwenEngine,
            )

            status = pool.reload(
                {"asr_provider": "qwen3", "qwen3_model_path": "", "live_asr_profile": "meeting"},
                {"enabled": True, "fuzzy_pinyin_enabled": True, "default_score": 5, "words": []},
            )

        self.assertIs(pool.live_engine, pool.final_engine)
        self.assertEqual(status["effective_provider"], "xasr")
        self.assertTrue(status["provider_fallback"])
        self.assertEqual(status["providers"]["qwen3"]["reason"], "model_not_configured")


if __name__ == "__main__":
    unittest.main()
