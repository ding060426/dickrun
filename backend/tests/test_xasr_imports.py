import subprocess
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class OptionalRuntimeImportTests(unittest.TestCase):
    def test_live_audio_can_import_without_sherpa_runtime(self):
        code = f"""
import builtins
import sys
sys.path.insert(0, {str(BACKEND_DIR)!r})
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'sherpa_onnx' or name.startswith('sherpa_onnx.'):
        raise ModuleNotFoundError('sherpa_onnx intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from xasr.contracts import ASRResult
from xasr.live_audio import LiveAudioSession
assert ASRResult(text='ok').text == 'ok'
assert LiveAudioSession.SAMPLE_RATE == 16000
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_main_can_import_when_xasr_engine_is_unavailable(self):
        code = f"""
import builtins
import sys
sys.path.insert(0, {str(BACKEND_DIR)!r})
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'xasr.asr_engine':
        raise ModuleNotFoundError('X-ASR engine intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import main
assert main.HAS_XASR is False
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
