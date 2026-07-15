import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import main
from diarization.contracts import DiarizationSegment
from diarization.pipeline import OfflineMeetingPipeline
from diarization.registry import MeetingRegistry
from xasr.contracts import ASRResult


class _Backend:
    provider_name = "test-diarization"

    def availability(self):
        return True, "ready"

    def diarize(self, audio, sample_rate, num_speakers=None):
        return [
            DiarizationSegment(0, 1, "SPEAKER_00", 0.9),
            DiarizationSegment(1, 2, "SPEAKER_01", 0.8),
        ]


class _Engine:
    is_model_available = True
    logic_validator = None

    def process_file(self, path, on_segment=None, on_progress=None, audio_buffer=None):
        return [ASRResult(text="一二三四", raw_text="一二三四", start_sec=0, end_sec=2)]

    def recognize_interval(
        self,
        audio_buffer,
        start_sec,
        end_sec,
        *,
        pre_padding_ms,
        post_padding_ms,
    ):
        text = "一二三" if start_sec < 0.5 else "一二三四"
        return ASRResult(text=text, raw_text=text, start_sec=start_sec, end_sec=end_sec)


class _Pool:
    def __init__(self, engine):
        self.engine = engine

    def create_final_session(self):
        return self.engine


class DiarizationApiTests(unittest.TestCase):
    def test_upload_returns_speakers_and_rename_updates_meeting(self):
        wav = io.BytesIO()
        sf.write(wav, np.zeros(32000, dtype=np.float32), 16000, format="WAV")
        engine = _Engine()
        pipeline = OfflineMeetingPipeline(_Backend())
        registry = MeetingRegistry()
        client = TestClient(main.app)

        with (
            patch.object(main, "xasr_engine", engine),
            patch.object(main, "xasr_pool", _Pool(engine)),
            patch.object(main, "meeting_pipeline", pipeline),
            patch.object(main, "meeting_registry", registry),
        ):
            response = client.post(
                "/api/audio/upload?file_id=api-test&enable_diarization=true&num_speakers=2",
                files={"file": ("meeting.wav", wav.getvalue(), "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["diarization"]["applied"])
            self.assertEqual(len(payload["speakers"]), 2)
            self.assertEqual(
                [segment["speaker_id"] for segment in payload["segments"]],
                ["SPEAKER_00", "SPEAKER_01"],
            )

            renamed = client.patch(
                "/api/meetings/api-test/speakers/SPEAKER_00",
                json={"name": "张三"},
            )
            self.assertEqual(renamed.status_code, 200)
            self.assertEqual(renamed.json()["segments"][0]["speaker_name"], "张三")


if __name__ == "__main__":
    unittest.main()
