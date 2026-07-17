import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_status_service import build_xasr_status


class _FakeEngine:
    is_model_available = True
    model_dir = "models/xasr"
    enable_endpoint_detection = False
    enable_logic_validation = True
    enable_hotword_correction = True
    enable_fuzzy_pinyin = True
    enable_uncertainty = True


class _FakePool:
    def status(self):
        return {
            "available_profiles": ["low-latency"],
            "selected_provider": "qwen3",
            "effective_provider": "xasr",
            "live_provider": "xasr",
            "provider_fallback": True,
            "provider_reason": "missing_dependencies",
            "providers": {
                "xasr": {"available": True, "reason": ""},
                "qwen3": {"available": False, "reason": "missing_dependencies", "missing": ["qwen_asr"]},
            },
            "live": {"requested_profile": "meeting", "effective_profile": "low-latency", "fallback": True, "chunk_ms": 160},
            "final": {"requested_profile": "meeting", "effective_profile": "low-latency", "fallback": True, "chunk_ms": 160},
            "shared_runtime": True,
            "file_vad_provider": "silero",
            "inference_threads": 12,
        }


class _FakePipeline:
    def status(self):
        return {"available": False, "reason": "missing_models"}


class ModelStatusServiceTests(unittest.TestCase):
    def test_build_status_preserves_legacy_and_adds_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            xasr = base / "models" / "xasr"
            vad = base / "models" / "vad"
            qwen = base / "models" / "qwen3"
            xasr.mkdir(parents=True)
            vad.mkdir(parents=True)
            qwen.mkdir(parents=True)
            (xasr / "tokens.txt").write_text("token 0\n", encoding="utf-8")
            for component in ("encoder", "decoder", "joiner"):
                (xasr / f"{component}-160ms.onnx").write_bytes(b"model")
            (vad / "silero_vad.onnx").write_bytes(b"model")
            (qwen / "config.json").write_text("{}", encoding="utf-8")
            (qwen / "model.safetensors").write_bytes(b"model")

            with patch.dict("os.environ", {"HUIWU_MODELS_DIR": str(base / "models")}, clear=True):
                status = build_xasr_status(
                    has_xasr=True,
                    xasr_engine=_FakeEngine(),
                    xasr_pool=_FakePool(),
                    xasr_loading=False,
                    runtime_settings={
                        "recognition": {
                            "asr_provider": "qwen3",
                            "qwen3_model_path": str(qwen),
                            "qwen3_device": "auto",
                            "qwen3_dtype": "auto",
                            "file_vad_threshold": 0.5,
                        },
                        "microphone": {"endpoint_grace_ms": 800},
                    },
                    hotword_settings={"active_count": 2},
                    meeting_pipeline=_FakePipeline(),
                    live_vad_model=vad / "silero_vad.onnx",
                    processing_workers=2,
                )

        self.assertTrue(status["available"])
        self.assertIn("paths", status)
        self.assertIn("profiles", status)
        self.assertEqual(status["providers"]["qwen3"]["mode"], "final_transcription_only")
        self.assertEqual(status["diarization"]["mode"], "asr_only")
        self.assertTrue(status["file_vad"]["available"])
        self.assertEqual(status["resources"]["processing_workers"], 2)


if __name__ == "__main__":
    unittest.main()
