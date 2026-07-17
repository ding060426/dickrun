import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr import model_paths


class ModelPathsTests(unittest.TestCase):
    def test_defaults_resolve_under_project_models(self):
        with patch.dict(os.environ, {}, clear=True):
            root = model_paths.resolve_project_root()

            self.assertEqual(model_paths.resolve_models_root(), root / "models")
            self.assertEqual(model_paths.resolve_xasr_model_dir(), root / "models" / "xasr")
            self.assertEqual(model_paths.resolve_qwen3_model_dir(), root / "models" / "qwen3")
            self.assertEqual(
                model_paths.resolve_vad_model_path(),
                root / "models" / "vad" / "silero_vad.onnx",
            )
            self.assertEqual(
                model_paths.resolve_diarization_segmentation_model(),
                root / "models" / "diarization" / "pyannote-segmentation-3.0.int8.onnx",
            )

    def test_explicit_model_env_overrides_models_root(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            with patch.dict(
                os.environ,
                {
                    "HUIWU_MODELS_DIR": str(base / "models-root"),
                    "DITING_XASR_MODEL_DIR": str(base / "custom-xasr"),
                    "DITING_QWEN3_MODEL_PATH": str(base / "custom-qwen"),
                    "DITING_SILERO_VAD_PATH": str(base / "vad.onnx"),
                },
                clear=True,
            ):
                self.assertEqual(model_paths.resolve_models_root(), base / "models-root")
                self.assertEqual(model_paths.resolve_xasr_model_dir(), base / "custom-xasr")
                self.assertEqual(model_paths.resolve_qwen3_model_dir(), base / "custom-qwen")
                self.assertEqual(model_paths.resolve_vad_model_path(), base / "vad.onnx")

    def test_inspect_xasr_profiles_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root)
            (model_dir / "tokens.txt").write_text("token 0\n", encoding="utf-8")
            (model_dir / "encoder-160ms.onnx").write_bytes(b"model")
            (model_dir / "decoder-160ms.onnx").write_bytes(b"model")
            (model_dir / "joiner-160ms.onnx").write_bytes(b"model")

            profiles = model_paths.inspect_xasr_profiles(model_dir)

        self.assertTrue(profiles["low-latency"]["complete"])
        self.assertFalse(profiles["meeting"]["complete"])
        self.assertIn("encoder", profiles["meeting"]["missing"])


if __name__ == "__main__":
    unittest.main()
