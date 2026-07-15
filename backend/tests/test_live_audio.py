import sys
import tempfile
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.asr_engine import ASRResult
from xasr.live_audio import (
    EnergyVad,
    LiveAudioSession,
    LiveAudioProtocolError,
    VadState,
    get_live_audio_profile,
    get_live_endpoint_grace_ms,
)
from xasr.recording import LiveRecording


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
        self.finalize_options = []

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

    def finalize_utterance(self, reset_stream=True, tail_pad_ms=0):
        self.finalize_calls.append(reset_stream)
        self.finalize_options.append((reset_stream, tail_pad_ms))
        return ASRResult(
            text="final utterance",
            raw_text="final utterance",
            is_partial=False,
            is_final=True,
        )


class LiveAudioSessionTests(unittest.TestCase):
    def test_energy_vad_starts_and_ends_after_configured_durations(self):
        vad = EnergyVad(
            threshold=0.01,
            min_speech_duration=0.1,
            min_silence_duration=0.1,
        )
        silence = np.zeros(1600, dtype=np.float32)
        speech = np.full(1600, 0.1, dtype=np.float32)

        self.assertFalse(vad.push(silence).is_speech)
        started = vad.push(speech)
        ended = vad.push(silence)

        self.assertTrue(started.speech_started)
        self.assertTrue(started.is_speech)
        self.assertTrue(ended.speech_ended)
        self.assertFalse(ended.is_speech)

    def test_metrics_include_protocol_and_latency_counters(self):
        engine = _FakeEngine()
        vad = _FakeVad([VadState(is_speech=True, speech_started=True)])
        session = LiveAudioSession(engine, vad=vad, pre_roll_ms=0)
        pcm = np.full(640, 1000, dtype="<i2").tobytes()

        session.push_binary_frame(struct.pack("<4sI", b"DTP2", 3) + pcm)
        session.finish()
        metrics = session.metrics()

        self.assertEqual(metrics["protocol_version"], 2)
        self.assertEqual(metrics["received_frames"], 1)
        self.assertEqual(metrics["last_sequence"], 3)
        self.assertEqual(metrics["partial_results"], 1)
        self.assertEqual(metrics["final_results"], 1)
        self.assertIsNotNone(metrics["first_partial_ms"])

    def test_versioned_binary_frames_report_sequence_gaps(self):
        engine = _FakeEngine()
        vad = _FakeVad([
            VadState(is_speech=True, speech_started=True),
            VadState(is_speech=True),
        ])
        session = LiveAudioSession(engine, vad=vad, pre_roll_ms=0)
        pcm = np.full(640, 1000, dtype="<i2").tobytes()

        session.push_binary_frame(struct.pack("<4sI", b"DTP2", 0) + pcm)
        session.push_binary_frame(struct.pack("<4sI", b"DTP2", 2) + pcm)

        self.assertEqual(session.last_sequence, 2)
        self.assertEqual(session.dropped_frames, 1)
        with self.assertRaises(LiveAudioProtocolError):
            session.push_binary_frame(struct.pack("<4sI", b"DTP2", 2) + pcm)

    def test_session_records_every_received_frame_and_publishes_on_stop(self):
        with tempfile.TemporaryDirectory() as root:
            engine = _FakeEngine()
            vad = _FakeVad([
                VadState(is_speech=False),
                VadState(is_speech=True, speech_started=True),
            ])
            recording = LiveRecording(root, "meeting-stream")
            session = LiveAudioSession(
                engine,
                vad=vad,
                recording=recording,
                pre_roll_ms=0,
            )
            frame = np.full(640, 1000, dtype="<i2").tobytes()

            session.push_pcm_s16le(frame)
            session.push_pcm_s16le(frame)
            session.finish()

            self.assertEqual(session.recording_result.received_samples, 1280)
            self.assertTrue(session.recording_result.path.is_file())

    def test_live_profiles_expose_distinct_endpoint_policies(self):
        meeting = get_live_audio_profile("meeting")
        dictation = get_live_audio_profile("dictation")

        self.assertEqual(meeting.pre_roll_ms, 700)
        self.assertGreater(meeting.endpoint_grace_ms, dictation.endpoint_grace_ms)
        self.assertEqual(meeting.tail_pad_ms, 1000)

    def test_stop_explicitly_tail_pads_before_final_result(self):
        engine = _FakeEngine()
        vad = _FakeVad([VadState(is_speech=True, speech_started=True)])
        session = LiveAudioSession(
            engine,
            vad=vad,
            pre_roll_ms=0,
            tail_pad_ms=1200,
        )
        frame = np.full(640, 1000, dtype="<i2").tobytes()
        session.push_pcm_s16le(frame)

        results = session.finish()

        self.assertEqual(engine.finalize_options, [(False, 1200)])
        self.assertEqual([item.text for item in results], ["final utterance"])

    def test_endpoint_grace_can_be_configured_from_environment(self):
        with patch.dict("os.environ", {"DITING_LIVE_ENDPOINT_GRACE_MS": "1200"}):
            self.assertEqual(get_live_endpoint_grace_ms(), 1200)

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

    def test_ungated_microphone_forwards_quiet_frames_before_vad_triggers(self):
        engine = _FakeEngine()
        vad = _FakeVad([
            VadState(is_speech=False),
            VadState(is_speech=False),
        ])
        session = LiveAudioSession(
            engine,
            vad=vad,
            pre_roll_ms=0,
            gate_audio=False,
        )
        frame = np.full(640, 50, dtype="<i2").tobytes()

        events = session.push_pcm_s16le(frame) + session.push_pcm_s16le(frame)

        self.assertEqual([len(chunk) for chunk in engine.chunks], [640, 640])
        self.assertEqual([item.text for item in events], ["partial-1", "partial-2"])
        self.assertFalse(session.metrics()["vad_gating"])

    def test_brief_pause_resumes_same_utterance_without_finalizing(self):
        engine = _FakeEngine()
        vad = _FakeVad([
            VadState(is_speech=True, speech_started=True),
            VadState(is_speech=False, speech_ended=True),
            VadState(is_speech=True, speech_started=True),
        ])
        session = LiveAudioSession(
            engine,
            vad=vad,
            pre_roll_ms=0,
            endpoint_grace_ms=800,
        )
        frame = np.full(640, 1000, dtype="<i2").tobytes()

        session.push_pcm_s16le(frame)
        pause_events = session.push_pcm_s16le(frame)
        resume_events = session.push_pcm_s16le(frame)

        self.assertEqual(engine.finalize_calls, [])
        self.assertEqual([len(chunk) for chunk in engine.chunks], [640, 640, 640])
        self.assertFalse(any(item.is_final for item in pause_events + resume_events))

    def test_sustained_pause_finalizes_after_endpoint_grace(self):
        engine = _FakeEngine()
        vad = _FakeVad([
            VadState(is_speech=True, speech_started=True),
            VadState(is_speech=False, speech_ended=True),
            VadState(is_speech=False),
            VadState(is_speech=False),
        ])
        session = LiveAudioSession(
            engine,
            vad=vad,
            pre_roll_ms=0,
            endpoint_grace_ms=80,
        )
        frame = np.full(640, 1000, dtype="<i2").tobytes()

        session.push_pcm_s16le(frame)
        session.push_pcm_s16le(frame)
        before_grace = session.push_pcm_s16le(frame)
        after_grace = session.push_pcm_s16le(frame)

        self.assertEqual(engine.finalize_calls, [True])
        self.assertFalse(any(item.is_final for item in before_grace))
        self.assertEqual([item.text for item in after_grace], ["partial-4", "final utterance"])
        self.assertTrue(after_grace[-1].is_final)


if __name__ == "__main__":
    unittest.main()
