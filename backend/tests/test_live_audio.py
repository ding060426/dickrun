import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.asr_engine import ASRResult
from xasr.live_audio import LiveAudioSession, VadState


class _FakeVad:
    def __init__(self, states):
        self.states = list(states)

    def push(self, samples):
        return self.states.pop(0)


class _FakeEngine:
    def __init__(self):
        self.started = False
        self.chunks = []
        self.finalize_calls = []

    def start_session(self):
        self.started = True

    def process_chunk(self, samples, sample_rate=16000):
        self.chunks.append(np.array(samples, copy=True))
        index = len(self.chunks)
        return ASRResult(
            text=f"partial-{index}",
            raw_text=f"partial-{index}",
            is_partial=True,
            is_final=False,
        )

    def finalize_utterance(self, reset_stream=True):
        self.finalize_calls.append(reset_stream)
        return ASRResult(
            text="final utterance",
            raw_text="final utterance",
            is_partial=False,
            is_final=True,
        )


class LiveAudioSessionTests(unittest.TestCase):
    def test_forwards_every_pcm_frame_once_speech_starts(self):
        engine = _FakeEngine()
        vad = _FakeVad([
            VadState(is_speech=True, speech_started=True),
            VadState(is_speech=True),
        ])
        session = LiveAudioSession(engine, vad=vad, pre_roll_ms=0)
        frame = np.full(640, 1000, dtype="<i2").tobytes()

        first = session.push_pcm_s16le(frame)
        second = session.push_pcm_s16le(frame)

        self.assertTrue(engine.started)
        self.assertEqual([len(chunk) for chunk in engine.chunks], [640, 640])
        self.assertEqual([item.text for item in first + second], ["partial-1", "partial-2"])

    def test_vad_speech_end_emits_final_result_and_resets_utterance(self):
        engine = _FakeEngine()
        vad = _FakeVad([
            VadState(is_speech=True, speech_started=True),
            VadState(is_speech=False, speech_ended=True),
        ])
        session = LiveAudioSession(engine, vad=vad, pre_roll_ms=0)
        frame = np.full(640, 1000, dtype="<i2").tobytes()

        session.push_pcm_s16le(frame)
        events = session.push_pcm_s16le(frame)

        self.assertEqual(engine.finalize_calls, [True])
        self.assertEqual([item.text for item in events], ["partial-2", "final utterance"])
        self.assertTrue(events[-1].is_final)


if __name__ == "__main__":
    unittest.main()
