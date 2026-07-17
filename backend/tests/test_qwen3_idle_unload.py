import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if "audio_buffer" not in sys.modules:
    fake_audio_buffer = types.ModuleType("audio_buffer")
    fake_audio_buffer.AudioBuffer = object
    fake_audio_buffer.load_audio_buffer = lambda *_args, **_kwargs: None
    sys.modules["audio_buffer"] = fake_audio_buffer

if "xasr.contracts" not in sys.modules:
    fake_contracts = types.ModuleType("xasr.contracts")
    fake_contracts.ASRResult = object
    sys.modules["xasr.contracts"] = fake_contracts

from xasr.qwen3_engine import Qwen3AsrEngine


class Qwen3IdleUnloadTests(unittest.TestCase):
    def test_runtime_status_and_idle_unload(self):
        with patch.dict(os.environ, {"DITING_QWEN3_IDLE_UNLOAD_SEC": "1"}):
            engine = Qwen3AsrEngine(model_path="model", model_loader=lambda *a, **k: object(), torch_module=object())
            engine._runtime["model"] = object()
            engine._runtime["last_used_at"] = time.time() - 2
            self.assertTrue(engine.runtime_status()["loaded"])
            self.assertTrue(engine.maybe_unload_idle())
            self.assertFalse(engine.runtime_status()["loaded"])

    def test_idle_unload_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            engine = Qwen3AsrEngine(model_path="model", model_loader=lambda *a, **k: object(), torch_module=object())
            engine._runtime["model"] = object()
            engine._runtime["last_used_at"] = time.time() - 999
            self.assertFalse(engine.maybe_unload_idle())


if __name__ == "__main__":
    unittest.main()
