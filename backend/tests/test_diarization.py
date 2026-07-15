import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from diarization.alignment import align_result, deduplicate_boundary
from diarization.contracts import DiarizationBackendResult, DiarizationSegment
from diarization.pipeline import OfflineMeetingPipeline
from diarization.registry import MeetingRegistry
from diarization.smoothing import smooth_timeline
from xasr.contracts import ASRResult


class _FakeBackend:
    provider_name = "test-speaker-backend"

    def __init__(self, *, available=True):
        self.available = available
        self.requested_speakers = None

    def availability(self):
        return self.available, "ready" if self.available else "models missing"

    def diarize(self, audio, sample_rate, num_speakers=None, on_progress=None):
        self.requested_speakers = num_speakers
        return [
            DiarizationSegment(0.0, 2.0, "SPEAKER_00", 0.9),
            DiarizationSegment(2.0, 4.0, "SPEAKER_01", 0.8),
        ]


class _ChunkOutcomeBackend(_FakeBackend):
    provider_name = "test-chunked-backend"

    def diarize(self, audio, sample_rate, num_speakers=None, on_progress=None):
        if on_progress:
            on_progress("diarization 1/2", 0.5)
            on_progress("diarization 2/2", 1.0)
        return DiarizationBackendResult(
            timeline=super().diarize(audio, sample_rate, num_speakers),
            metadata={
                "chunked": True,
                "chunk_count": 2,
                "worker_count": 2,
                "skipped_silence_sec": 12.5,
            },
        )


class _FailingBackend(_FakeBackend):
    provider_name = "failing-diarization"

    def diarize(self, audio, sample_rate, num_speakers=None, on_progress=None):
        raise RuntimeError("chunk 2 failed")


class _FakeEngine:
    def __init__(self):
        self.shared_audio = None

    def process_file(self, path, on_segment=None, on_progress=None, audio_buffer=None):
        self.shared_audio = audio_buffer
        return [ASRResult(text="项目计划", raw_text="项目计划", start_sec=0, end_sec=4)]

    def recognize_interval(
        self,
        audio_buffer,
        start_sec,
        end_sec,
        *,
        pre_padding_ms,
        post_padding_ms,
    ):
        text = "项目计划" if start_sec < 1 else "目计划确认"
        return ASRResult(text=text, raw_text=text, start_sec=start_sec, end_sec=end_sec)


class _LossyBoundaryEngine(_FakeEngine):
    def process_file(self, path, on_segment=None, on_progress=None, audio_buffer=None):
        self.shared_audio = audio_buffer
        text = "这是一段完整且不能丢失的会议文本"
        return [ASRResult(text=text, raw_text=text, start_sec=0, end_sec=4)]

    def recognize_interval(
        self,
        audio_buffer,
        start_sec,
        end_sec,
        *,
        pre_padding_ms,
        post_padding_ms,
    ):
        text = "甲" if start_sec < 1 else "乙"
        return ASRResult(text=text, raw_text=text, start_sec=start_sec, end_sec=end_sec)


class DiarizationTests(unittest.TestCase):
    def test_alignment_uses_total_overlap_and_marks_real_overlap(self):
        result = ASRResult(start_sec=0, end_sec=4)
        timeline = [
            DiarizationSegment(0, 3, "SPEAKER_00", 0.9),
            DiarizationSegment(
                2,
                4,
                "SPEAKER_01",
                0.8,
                overlap=True,
                overlap_speakers=("SPEAKER_00", "SPEAKER_01"),
            ),
        ]

        align_result(result, timeline)

        self.assertEqual(result.speaker_id, "SPEAKER_00")
        self.assertTrue(result.overlap)
        self.assertEqual(result.overlap_speakers, ["SPEAKER_00", "SPEAKER_01"])

    def test_smoothing_does_not_merge_across_another_speaker(self):
        timeline = smooth_timeline(
            [
                DiarizationSegment(0.0, 1.0, "SPEAKER_00"),
                DiarizationSegment(1.0, 1.2, "SPEAKER_01"),
                DiarizationSegment(1.2, 2.0, "SPEAKER_00"),
            ],
            merge_gap_sec=0.5,
        )
        self.assertEqual(len(timeline), 3)

    def test_pipeline_redecodes_only_at_speaker_boundaries(self):
        backend = _FakeBackend()
        engine = _FakeEngine()
        pipeline = OfflineMeetingPipeline(backend)
        delivered = []
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "meeting.wav"
            sf.write(path, np.zeros(4 * 16000, dtype=np.float32), 16000)
            run = pipeline.process_file(
                path,
                engine,
                num_speakers=2,
                on_segment=lambda result, index, total: delivered.append(
                    (result, index, total)
                ),
            )

        self.assertIsNotNone(engine.shared_audio)
        self.assertEqual(backend.requested_speakers, 2)
        self.assertEqual([result.speaker_id for result in run.results], [
            "SPEAKER_00",
            "SPEAKER_01",
        ])
        self.assertEqual(run.results[1].text, "确认")
        self.assertEqual(len(delivered), 2)
        self.assertTrue(run.applied)

    def test_boundary_redecode_keeps_base_text_when_candidates_lose_words(self):
        engine = _LossyBoundaryEngine()
        pipeline = OfflineMeetingPipeline(_FakeBackend())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "meeting.wav"
            sf.write(path, np.zeros(4 * 16000, dtype=np.float32), 16000)
            run = pipeline.process_file(path, engine, num_speakers=2)

        self.assertEqual(len(run.results), 1)
        self.assertEqual(run.results[0].text, "这是一段完整且不能丢失的会议文本")
        self.assertEqual(run.boundary_redecoded_segments, 0)

    def test_unavailable_backend_falls_back_without_failing_asr(self):
        engine = _FakeEngine()
        pipeline = OfflineMeetingPipeline(_FakeBackend(available=False))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "meeting.wav"
            sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
            run = pipeline.process_file(path, engine)
        self.assertFalse(run.applied)
        self.assertEqual(run.reason, "models missing")
        self.assertEqual(len(run.results), 1)

    def test_pipeline_forwards_chunk_progress_and_metadata(self):
        progress = []
        pipeline = OfflineMeetingPipeline(_ChunkOutcomeBackend())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "meeting.wav"
            sf.write(path, np.zeros(4 * 16000, dtype=np.float32), 16000)
            run = pipeline.process_file(
                path,
                _FakeEngine(),
                on_progress=lambda stage, fraction: progress.append((stage, fraction)),
            )

        self.assertIn(("diarization 1/2", 0.2), progress)
        self.assertIn(("diarization 2/2", 0.35), progress)
        self.assertTrue(run.metadata()["chunked"])
        self.assertEqual(run.metadata()["chunk_count"], 2)

    def test_runtime_diarization_failure_falls_back_to_complete_asr(self):
        engine = _FakeEngine()
        pipeline = OfflineMeetingPipeline(_FailingBackend())
        progress = []
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "meeting.wav"
            sf.write(path, np.zeros(4 * 16000, dtype=np.float32), 16000)
            run = pipeline.process_file(
                path,
                engine,
                on_progress=lambda stage, fraction: progress.append((stage, fraction)),
            )

        self.assertFalse(run.applied)
        self.assertIn("chunk 2 failed", run.reason)
        self.assertEqual(len(run.results), 1)
        self.assertIsNotNone(engine.shared_audio)
        self.assertEqual(progress[-1], ("done", 1.0))

    def test_boundary_deduplication_requires_three_characters(self):
        self.assertEqual(deduplicate_boundary("项目计划", "目计划确认"), "确认")
        self.assertEqual(deduplicate_boundary("你好", "你好"), "你好")

    def test_speaker_rename_updates_all_retained_segments(self):
        registry = MeetingRegistry()
        registry.register(
            "meeting-1",
            filename="meeting.wav",
            speakers=[{"id": "SPEAKER_00", "name": None, "duration": 3}],
            segments=[
                {
                    "speaker_id": "SPEAKER_00",
                    "speaker_name": None,
                    "audio_wav_base64": "large-audio",
                }
            ],
        )
        updated = registry.rename("meeting-1", "SPEAKER_00", "张三")
        self.assertEqual(updated["speakers"][0]["name"], "张三")
        self.assertEqual(updated["segments"][0]["speaker_name"], "张三")
        self.assertNotIn("audio_wav_base64", updated["segments"][0])


if __name__ == "__main__":
    unittest.main()
