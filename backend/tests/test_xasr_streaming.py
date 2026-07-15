import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from xasr.asr_engine import ASRResult, XASREngine


class _FakeStreamingAsr:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def get_final_result(self):
        return "实时识别结果"

    def input_finished(self):
        pass


class _SilentStreamingAsr:
    def accept_waveform(self, samples, sample_rate=16000):
        pass

    def decode(self):
        return 0

    def get_partial_result(self):
        return ""

    def is_endpoint(self):
        return False

    def input_finished(self):
        pass


class _FakeRuntime:
    def __init__(self):
        self.sessions = []

    def create_session(self):
        session = _FakeStreamingAsr()
        self.sessions.append(session)
        return session


class _FakeFileSegmenter:
    provider_name = "test-silero-file-vad"

    def __init__(self):
        self.calls = []

    def detect(self, audio, sample_rate):
        self.calls.append((len(audio), sample_rate))
        return [(0.25, 0.75)]


class _TimelineFileSegmenter:
    provider_name = "test-timeline-vad"

    def detect(self, audio, sample_rate):
        return [(0.0, 2.0), (5.0, 7.0), (9.0, 11.0)]


class _CumulativeStreamingAsr:
    """Fake a recognizer whose missing short phrase only appears with context."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.accepted_samples = 0

    def accept_waveform(self, samples, sample_rate=16000):
        self.accepted_samples += len(samples)

    def decode(self):
        return 1

    def get_partial_result(self):
        seconds = self.accepted_samples / self.sample_rate
        if seconds >= 11.0:
            return "第一句。被吞掉的短句。第三句。"
        if seconds >= 7.0:
            return "第一句。被吞掉的短句。"
        if seconds >= 2.0:
            return "第一句。"
        return ""

    def input_finished(self):
        pass

    def get_final_result(self):
        return "第一句。被吞掉的短句。第三句。"


def _write_model_files(model_dir: str, chunk_ms: int = 160) -> None:
    root = Path(model_dir)
    for name in (
        "tokens.txt",
        f"encoder-{chunk_ms}ms.onnx",
        f"decoder-{chunk_ms}ms.onnx",
        f"joiner-{chunk_ms}ms.onnx",
    ):
        (root / name).write_bytes(b"test")


class XASRStreamingSessionTests(unittest.TestCase):
    def test_final_result_exposes_text_cleanup_metadata(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(
                model_dir=model_dir,
                enable_logic_validation=False,
                enable_uncertainty=False,
            )

        result = engine._build_result(
            raw_text="嗯，就是说好好好我们开始开会啊",
            is_partial=False,
            is_final=True,
        )

        self.assertTrue(result.postprocessed)
        self.assertIn("就是说", result.fillers_removed)
        self.assertIn("好好好", result.repetitions_merged)

    def test_silent_partial_uses_fast_path_and_samples_acoustics_periodically(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)
        engine.asr = _SilentStreamingAsr()
        engine._session_active = True
        chunk = np.zeros(640, dtype=np.float32)

        with (
            patch("xasr.asr_engine.estimate_snr", return_value=20.0) as snr,
            patch("xasr.asr_engine.estimate_rt60", return_value=0.2) as rt60,
        ):
            results = [engine.process_chunk(chunk) for _ in range(6)]

        self.assertTrue(all(result.text == "" for result in results))
        self.assertEqual(snr.call_count, 2)
        self.assertEqual(rt60.call_count, 2)

    def test_end_session_releases_stream_reference(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)
        engine.asr = _SilentStreamingAsr()
        engine._session_active = True

        engine.end_session()

        self.assertIsNone(engine.asr)

    def test_empty_recording_finishes_without_dividing_by_zero(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)
            engine._load_audio = lambda _: (np.empty(0, dtype=np.float32), 16000)

            results = engine.process_file("empty.wav")

        self.assertEqual(results, [])

    def test_file_processing_uses_configured_model_segmenter(self):
        with tempfile.TemporaryDirectory() as model_dir:
            segmenter = _FakeFileSegmenter()
            engine = XASREngine(model_dir=model_dir, file_segmenter=segmenter)
            engine._load_audio = lambda _: (np.zeros(16000, dtype=np.float32), 16000)
            engine._process_speech_segment = lambda audio, sr, start, end: ASRResult(
                text="speech",
                raw_text="speech",
                is_final=True,
                start_sec=start,
                end_sec=end,
            )

            results = engine.process_file("meeting.wav")

        self.assertEqual(segmenter.calls, [(16000, 16000)])
        self.assertEqual([(item.start_sec, item.end_sec) for item in results], [(0.25, 0.75)])
        self.assertEqual(engine.file_vad_provider, "test-silero-file-vad")

    def test_file_processing_preserves_context_across_vad_regions(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _write_model_files(model_dir, chunk_ms=960)
            engine = XASREngine(
                model_dir=model_dir,
                asr_profile="meeting",
                file_segmenter=_TimelineFileSegmenter(),
                enable_logic_validation=False,
                enable_hotword_correction=False,
                enable_uncertainty=False,
                enable_text_postprocess=False,
            )
            engine._load_audio = lambda _: (np.zeros(12 * 16000, dtype=np.float32), 16000)
            engine._create_asr = lambda: _CumulativeStreamingAsr()

            # This models the old behavior: an isolated quiet VAD region decodes empty.
            engine._process_speech_segment = lambda audio, sr, start, end: (
                None
                if start == 5.0
                else ASRResult(
                    text="第一句。" if start == 0.0 else "第三句。",
                    raw_text="第一句。" if start == 0.0 else "第三句。",
                    is_final=True,
                    start_sec=start,
                    end_sec=end,
                )
            )

            results = engine.process_file("meeting.wav")

        transcript = "".join(item.raw_text for item in results)
        self.assertIn("被吞掉的短句", transcript)
        self.assertEqual(results[0].start_sec, 0.0)
        self.assertEqual(results[-1].end_sec, 11.0)

    def test_meeting_profile_selects_960ms_model_files(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _write_model_files(model_dir, chunk_ms=960)
            engine = XASREngine(model_dir=model_dir, asr_profile="meeting")

        self.assertTrue(engine.is_model_available)
        self.assertEqual(engine.asr_profile, "meeting")
        self.assertEqual(engine.chunk_ms, 960)

    def test_forked_engines_share_runtime_but_open_independent_sessions(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as model_dir:
            _write_model_files(model_dir)
            engine = XASREngine(model_dir=model_dir, recognizer_runtime=runtime)
            forked = engine.fork_session()

        engine.start_session()
        forked.start_session()

        self.assertIsNot(engine.asr, forked.asr)
        self.assertEqual(runtime.sessions, [engine.asr, forked.asr])
        self.assertIsNot(engine.logic_validator, forked.logic_validator)

    def test_runtime_affecting_fork_override_does_not_reuse_old_runtime(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as model_dir:
            _write_model_files(model_dir)
            _write_model_files(model_dir, chunk_ms=960)
            engine = XASREngine(model_dir=model_dir, recognizer_runtime=runtime)
            forked = engine.fork_session(asr_profile="meeting")

        self.assertIsNone(forked._recognizer_runtime)
        self.assertEqual(forked.chunk_ms, 960)

    def test_start_session_lazily_creates_streaming_recognizer(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)

        recognizer = _FakeStreamingAsr()
        engine.model_available = True
        engine._create_asr = lambda: recognizer

        engine.start_session()

        self.assertIs(engine.asr, recognizer)
        self.assertTrue(engine._session_active)
        self.assertEqual(recognizer.reset_calls, 0)

    def test_finalize_utterance_returns_final_result_and_opens_next_stream(self):
        with tempfile.TemporaryDirectory() as model_dir:
            engine = XASREngine(model_dir=model_dir)

        recognizer = _FakeStreamingAsr()
        engine.asr = recognizer
        engine._session_active = True

        result = engine.finalize_utterance(reset_stream=True)

        self.assertEqual(result.raw_text, "实时识别结果")
        self.assertTrue(result.is_final)
        self.assertEqual(recognizer.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
