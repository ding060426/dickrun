import base64
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules import record_store


class RecordStoreTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(self.db_path)
        self.original_db_path = record_store.DB_PATH
        record_store.DB_PATH = Path(self.db_path)
        record_store.init_db()

    def tearDown(self):
        record_store.DB_PATH = self.original_db_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_saves_searchable_full_text_and_segment_audio_blob(self):
        wav_bytes = b"RIFF" + bytes(range(32))
        audio_base64 = base64.b64encode(wav_bytes).decode("ascii")

        saved = record_store.save_record(
            {
                "title": "周会记录",
                "source_type": "microphone",
                "source_filename": "mic-session.wav",
                "source_mime_type": "audio/wav",
                "source_size_bytes": 4096,
                "meeting_id": "meeting-1",
                "created_by": "user-1",
                "speakers": [{"id": "SPEAKER_00", "name": "张三"}],
                "segments": [
                    {
                        "index": 1,
                        "speaker_id": "SPEAKER_00",
                        "speaker_name": "张三",
                        "start_sec": 0,
                        "end_sec": 2.5,
                        "text": "项目按计划推进",
                        "audio_wav_base64": audio_base64,
                        "asr_confidence": 0.93,
                    }
                ],
            }
        )

        self.assertIn("张三", saved["full_text"])
        self.assertIn("项目按计划推进", saved["full_text"])
        self.assertEqual(saved["segments"][0]["audio_wav_base64"], audio_base64)
        self.assertEqual(saved["segments"][0]["speaker_id"], "SPEAKER_00")

        listed = record_store.list_records(user_id="user-1", q="计划")
        self.assertEqual(listed["total"], 1)
        self.assertEqual(listed["items"][0]["source_type"], "microphone")
        self.assertNotIn("segments", listed["items"][0])

        conn = sqlite3.connect(self.db_path)
        try:
            blob = conn.execute(
                "SELECT audio_blob FROM meeting_record_segments"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(blob, wav_bytes)

    def test_updates_record_without_losing_creation_time_and_can_delete_it(self):
        saved = record_store.save_record(
            {
                "title": "初稿",
                "created_by": "user-1",
                "segments": [{"text": "第一版", "start_sec": 0, "end_sec": 1}],
            }
        )
        updated = record_store.save_record(
            {
                "title": "定稿",
                "created_by": "user-1",
                "segments": [{"text": "第二版", "start_sec": 0, "end_sec": 1}],
            },
            record_id=saved["id"],
        )

        self.assertEqual(updated["id"], saved["id"])
        self.assertEqual(updated["created_at"], saved["created_at"])
        self.assertEqual(updated["title"], "定稿")
        self.assertTrue(record_store.delete_record(saved["id"]))
        self.assertIsNone(record_store.get_record(saved["id"]))


if __name__ == "__main__":
    unittest.main()
