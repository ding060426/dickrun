import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from audio_buffer import AudioBuffer
from xasr.qwen3_engine import Qwen3AsrEngine


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(text="这是 Qwen3 的转写结果")]


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"
    cuda = SimpleNamespace(is_available=lambda: True)


class Qwen3AsrEngineTests(unittest.TestCase):
    def test_file_transcription_reuses_model_and_passes_hotwords_as_context(self):
        model = _FakeModel()
        load_calls = []

        def load_model(path, **kwargs):
            load_calls.append((path, kwargs))
            return model

        engine = Qwen3AsrEngine(
            model_path="Qwen/Qwen3-ASR-0.6B",
            device="auto",
            dtype="auto",
            hotwords=["滴听", "会议纪要"],
            hotword_scores={"会议纪要": 9, "滴听": 5},
            model_loader=load_model,
            torch_module=_FakeTorch,
        ).warmup()
        audio = AudioBuffer(np.ones(16000, dtype=np.float32) * 0.01)

        results = engine.process_file("unused.wav", audio_buffer=audio)
        second = engine.fork_session().recognize_interval(audio, 0.1, 0.8)

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1]["device_map"], "cuda:0")
        self.assertEqual(load_calls[0][1]["dtype"], "bf16")
        self.assertEqual(results[0].text, "这是 Qwen3 的转写结果")
        self.assertTrue(results[0].is_final)
        self.assertEqual(second.start_sec, 0.1)
        self.assertIn("会议纪要、滴听", model.calls[0]["context"])
        self.assertEqual(model.calls[0]["audio"][1], 16000)


if __name__ == "__main__":
    unittest.main()
