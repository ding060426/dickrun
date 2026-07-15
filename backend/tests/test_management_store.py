import sys
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules import meeting_db
from modules import management_store
from modules.management_store import select_management_store


class ManagementStoreTests(unittest.TestCase):
    def test_uses_local_sqlite_without_complete_supabase_configuration(self):
        self.assertIs(select_management_store({}), meeting_db)
        self.assertIs(
            select_management_store({"SUPABASE_URL": "https://example.supabase.co"}),
            meeting_db,
        )

    def test_default_selection_loads_complete_supabase_config_from_backend_env(self):
        handle, env_path = tempfile.mkstemp(suffix=".env")
        os.close(handle)
        Path(env_path).write_text(
            "SUPABASE_URL=https://example.supabase.co\nSUPABASE_KEY=test-key\n",
            encoding="utf-8",
        )
        fake_supabase = types.ModuleType("modules.supabase_db")
        try:
            with (
                patch.object(management_store, "BACKEND_ENV_PATH", Path(env_path)),
                patch.dict(os.environ, {}, clear=True),
                patch.dict(sys.modules, {"modules.supabase_db": fake_supabase}),
            ):
                self.assertIs(select_management_store(), fake_supabase)
        finally:
            if os.path.exists(env_path):
                os.unlink(env_path)

    def test_local_store_persists_transcription_analysis(self):
        handle, db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(db_path)
        original_data_dir = meeting_db.DATA_DIR
        original_db_path = meeting_db.DB_PATH
        meeting_db.DATA_DIR = Path(db_path).parent
        meeting_db.DB_PATH = Path(db_path)
        try:
            meeting_db.init_db()
            saved = meeting_db.save_analysis(
                {
                    "title": "Weekly sync.wav",
                    "transcript_json": [
                        {
                            "text": "项目按计划推进",
                            "speaker": "SPEAKER_00",
                            "start": 0.0,
                            "end": 2.5,
                        }
                    ],
                    "segments_count": 1,
                    "duration_sec": 2.5,
                }
            )

            loaded = meeting_db.get_analysis(saved["id"])
            self.assertEqual(loaded["title"], "Weekly sync.wav")
            self.assertEqual(loaded["transcript_json"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(meeting_db.list_analyses()[0]["segments_count"], 1)
        finally:
            meeting_db.DATA_DIR = original_data_dir
            meeting_db.DB_PATH = original_db_path
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
