import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from diarization.chunked_backend import (
    ChunkedDiarizationBackend,
    ChunkedDiarizationConfig,
)
from diarization.contracts import DiarizationSegment


class _RecordingBackend:
    provider_name = "recording-backend"

    def __init__(self):
        self.calls = []

    def availability(self):
        return True, "ready"

    def diarize(self, audio, sample_rate, num_speakers=None):
        self.calls.append((len(audio), sample_rate, num_speakers))
        return [
            DiarizationSegment(
                start_sec=0.0,
                end_sec=len(audio) / sample_rate,
                speaker_id="SPEAKER_00",
                confidence=0.9,
            )
        ]


class _NamedSpeakerBackend(_RecordingBackend):
    def __init__(self, speaker_id):
        super().__init__()
        self.speaker_id = speaker_id

    def diarize(self, audio, sample_rate, num_speakers=None):
        self.calls.append((len(audio), sample_rate, num_speakers))
        return [
            DiarizationSegment(
                0.0,
                len(audio) / sample_rate,
                self.speaker_id,
                confidence=0.9,
            )
        ]


class _SequenceSpeakerBackend(_RecordingBackend):
    def __init__(self, speaker_ids):
        super().__init__()
        self.speaker_ids = speaker_ids

    def diarize(self, audio, sample_rate, num_speakers=None):
        speaker_id = next(self.speaker_ids)
        self.calls.append((len(audio), sample_rate, num_speakers))
        return [
            DiarizationSegment(
                0.0,
                len(audio) / sample_rate,
                speaker_id,
                confidence=0.9,
            )
        ]


class _MeanSpeakerEmbedder:
    def extract(self, audio, sample_rate):
        mean = float(np.mean(audio)) if len(audio) else 0.0
        return np.asarray([1.0, mean], dtype=np.float32)


class _SelectiveSpeakerEmbedder(_MeanSpeakerEmbedder):
    def extract(self, audio, sample_rate):
        if float(np.mean(audio)) < 0:
            return np.zeros(2, dtype=np.float32)
        return super().extract(audio, sample_rate)


class _StaticSpeechDetector:
    def __init__(self, regions):
        self.regions = regions

    def detect(self, audio, sample_rate):
        return list(self.regions)


class _ConcurrencyProbeBackend(_RecordingBackend):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def diarize(self, audio, sample_rate, num_speakers=None):
        deadline = time.monotonic() + 0.5
        with self.state["condition"]:
            self.state["active"] += 1
            self.state["max_active"] = max(
                self.state["max_active"],
                self.state["active"],
            )
            if self.state["active"] >= 2:
                self.state["overlapped"] = True
            self.state["condition"].notify_all()
            while not self.state["overlapped"] and time.monotonic() < deadline:
                self.state["condition"].wait(deadline - time.monotonic())
            overlapped = self.state["overlapped"]
            self.state["active"] -= 1
            self.state["condition"].notify_all()
        if not overlapped:
            raise AssertionError("chunk inference did not overlap")
        return super().diarize(audio, sample_rate, num_speakers)


class _TransientFailureBackend(_RecordingBackend):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def diarize(self, audio, sample_rate, num_speakers=None):
        self.state["calls"] += 1
        if self.state["calls"] == 1:
            raise RuntimeError("temporary worker failure")
        return super().diarize(audio, sample_rate, num_speakers)


class _AlwaysFailingBackend(_RecordingBackend):
    def diarize(self, audio, sample_rate, num_speakers=None):
        raise RuntimeError("persistent worker failure")


class _BoundaryArtifactBackend(_RecordingBackend):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def diarize(self, audio, sample_rate, num_speakers=None):
        self.state["calls"] += 1
        if self.state["calls"] == 1:
            return super().diarize(audio, sample_rate, num_speakers)
        return [DiarizationSegment(0.95, 1.05, "SPEAKER_09", confidence=0.9)]


class ChunkedDiarizationTests(unittest.TestCase):
    def test_short_audio_preserves_the_existing_whole_file_path(self):
        base = _RecordingBackend()
        backend = ChunkedDiarizationBackend(
            base,
            config=ChunkedDiarizationConfig(long_audio_threshold_sec=10.0),
        )
        audio = np.zeros(4 * 16000, dtype=np.float32)

        timeline = backend.diarize(audio, 16000, num_speakers=2).timeline

        self.assertEqual(base.calls, [(len(audio), 16000, 2)])
        self.assertEqual(timeline, [
            DiarizationSegment(0.0, 4.0, "SPEAKER_00", confidence=0.9)
        ])

    def test_long_audio_splits_at_long_silence_and_restores_global_time(self):
        whole_file = _RecordingBackend()
        workers = []

        def worker_factory():
            worker = _RecordingBackend()
            workers.append(worker)
            return worker

        backend = ChunkedDiarizationBackend(
            whole_file,
            worker_factory=worker_factory,
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=1,
            ),
        )
        audio = np.zeros(12 * 16000, dtype=np.float32)

        timeline = backend.diarize(audio, 16000).timeline

        self.assertEqual(whole_file.calls, [])
        self.assertEqual([call[0] for worker in workers for call in worker.calls], [
            6 * 16000,
            6 * 16000,
        ])
        self.assertEqual(
            [(segment.start_sec, segment.end_sec, segment.speaker_id) for segment in timeline],
            [(0.0, 5.0, "SPEAKER_00"), (7.0, 12.0, "SPEAKER_00")],
        )

    def test_same_voice_gets_one_global_speaker_across_chunks(self):
        local_speakers = iter(["SPEAKER_04", "SPEAKER_01"])
        backend = ChunkedDiarizationBackend(
            _RecordingBackend(),
            worker_factory=lambda: _SequenceSpeakerBackend(local_speakers),
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=1,
                stitch_threshold=0.99,
            ),
        )
        audio = np.zeros(12 * 16000, dtype=np.float32)
        audio[: 5 * 16000] = 0.25
        audio[7 * 16000 :] = 0.25

        timeline = backend.diarize(audio, 16000).timeline

        self.assertEqual(
            [segment.speaker_id for segment in timeline],
            ["SPEAKER_00", "SPEAKER_00"],
        )

    def test_long_audio_chunks_run_with_bounded_parallelism(self):
        state = {
            "condition": threading.Condition(),
            "active": 0,
            "max_active": 0,
            "overlapped": False,
        }
        backend = ChunkedDiarizationBackend(
            _RecordingBackend(),
            worker_factory=lambda: _ConcurrencyProbeBackend(state),
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=2,
            ),
        )

        backend.diarize(np.zeros(12 * 16000, dtype=np.float32), 16000)

        self.assertEqual(state["max_active"], 2)

    def test_known_speaker_count_is_applied_globally_not_to_every_chunk(self):
        workers = []

        def worker_factory():
            worker = _RecordingBackend()
            workers.append(worker)
            return worker

        backend = ChunkedDiarizationBackend(
            _RecordingBackend(),
            worker_factory=worker_factory,
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=1,
            ),
        )
        audio = np.zeros(12 * 16000, dtype=np.float32)
        audio[: 5 * 16000] = 0.25
        audio[7 * 16000 :] = -0.25

        timeline = backend.diarize(audio, 16000, num_speakers=2).timeline

        self.assertEqual(
            [call[2] for worker in workers for call in worker.calls],
            [None, None],
        )
        self.assertEqual(
            [segment.speaker_id for segment in timeline],
            ["SPEAKER_00", "SPEAKER_01"],
        )

    def test_chunk_progress_and_stats_are_returned_per_run(self):
        progress = []
        backend = ChunkedDiarizationBackend(
            _RecordingBackend(),
            worker_factory=_RecordingBackend,
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=1,
            ),
        )

        outcome = backend.diarize(
            np.zeros(12 * 16000, dtype=np.float32),
            16000,
            on_progress=lambda stage, fraction: progress.append((stage, fraction)),
        )

        self.assertEqual(outcome.metadata["chunk_count"], 2)
        self.assertEqual(outcome.metadata["worker_count"], 1)
        self.assertEqual(outcome.metadata["skipped_silence_sec"], 2.0)
        self.assertEqual(progress[-1], ("diarization 2/2", 1.0))

    def test_transient_chunk_failure_is_retried_with_a_fresh_worker(self):
        state = {"calls": 0}
        backend = ChunkedDiarizationBackend(
            _RecordingBackend(),
            worker_factory=lambda: _TransientFailureBackend(state),
            speech_detector=_StaticSpeechDetector([(0.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=20.0,
                max_chunk_sec=20.0,
                max_workers=1,
            ),
        )

        outcome = backend.diarize(
            np.full(12 * 16000, 0.25, dtype=np.float32),
            16000,
        )

        self.assertEqual(state["calls"], 2)
        self.assertEqual(len(outcome.timeline), 1)

    def test_persistent_chunk_failure_falls_back_to_whole_file_diarization(self):
        whole_file = _RecordingBackend()
        backend = ChunkedDiarizationBackend(
            whole_file,
            worker_factory=_AlwaysFailingBackend,
            speech_detector=_StaticSpeechDetector([(0.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=20.0,
                max_chunk_sec=20.0,
                max_workers=1,
            ),
        )

        outcome = backend.diarize(np.zeros(12 * 16000, dtype=np.float32), 16000)

        self.assertEqual(len(whole_file.calls), 1)
        self.assertFalse(outcome.metadata["chunked"])
        self.assertIn("persistent worker failure", outcome.metadata["chunk_fallback"])
        self.assertEqual(len(outcome.timeline), 1)

    def test_chunk_boundary_does_not_emit_tiny_speaker_artifacts(self):
        state = {"calls": 0}
        backend = ChunkedDiarizationBackend(
            _RecordingBackend(),
            worker_factory=lambda: _BoundaryArtifactBackend(state),
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_MeanSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=1,
                min_output_segment_sec=0.1,
            ),
        )

        outcome = backend.diarize(np.zeros(12 * 16000, dtype=np.float32), 16000)

        self.assertEqual(len(outcome.timeline), 1)

    def test_known_count_falls_back_when_a_chunk_has_no_usable_voiceprint(self):
        whole_file = _RecordingBackend()
        backend = ChunkedDiarizationBackend(
            whole_file,
            worker_factory=_RecordingBackend,
            speech_detector=_StaticSpeechDetector([(0.0, 5.0), (7.0, 12.0)]),
            speaker_embedder=_SelectiveSpeakerEmbedder(),
            config=ChunkedDiarizationConfig(
                long_audio_threshold_sec=10.0,
                target_chunk_sec=6.0,
                max_chunk_sec=8.0,
                overlap_sec=1.0,
                skip_silence_sec=1.5,
                max_workers=1,
            ),
        )
        audio = np.zeros(12 * 16000, dtype=np.float32)
        audio[: 5 * 16000] = 0.25
        audio[7 * 16000 :] = -0.25

        outcome = backend.diarize(audio, 16000, num_speakers=1)

        self.assertEqual(len(whole_file.calls), 1)
        self.assertFalse(outcome.metadata["chunked"])
        self.assertIn("voiceprint", outcome.metadata["chunk_fallback"])



if __name__ == "__main__":
    unittest.main()
