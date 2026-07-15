import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from modules import meeting_db


class ManagementApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(self.db_path)
        self.original_data_dir = meeting_db.DATA_DIR
        self.original_db_path = meeting_db.DB_PATH
        meeting_db.DATA_DIR = Path(self.db_path).parent
        meeting_db.DB_PATH = Path(self.db_path)
        meeting_db.init_db()
        main.db = meeting_db

    def tearDown(self):
        meeting_db.DATA_DIR = self.original_data_dir
        meeting_db.DB_PATH = self.original_db_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    async def test_user_can_attach_a_transcription_to_a_managed_meeting(self):
        registered = await main.auth_register(
            {"username": "owner", "password": "secret123"}
        )
        self.assertEqual(registered["user"]["username"], "owner")

        login = await main.auth_login(
            {"username": "owner", "password": "secret123"}
        )
        authorization = f"Bearer {login['token']}"

        reservation = await main.api_create_reservation(
            {
                "title": "Weekly sync",
                "start_time": "2026-07-16T09:00:00",
                "end_time": "2026-07-16T10:00:00",
            },
            authorization,
        )
        meeting_id = reservation["reservation"]["id"]

        saved = await main.api_save_analysis(
            {
                "meeting_id": meeting_id,
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
            },
            authorization,
        )
        self.assertEqual(saved["analysis"]["segments_count"], 1)

        listed = await main.api_list_analyses(None, authorization)
        self.assertEqual(listed["analyses"][0]["meeting_id"], meeting_id)
        self.assertEqual(
            listed["analyses"][0]["transcript_json"][0]["speaker"],
            "SPEAKER_00",
        )
        reservations = await main.api_list_reservations(
            None,
            None,
            authorization,
        )
        self.assertNotEqual(reservations["reservations"][0]["time_status"], "未知")

        await main.auth_register({"username": "other", "password": "secret123"})
        other_login = await main.auth_login(
            {"username": "other", "password": "secret123"}
        )
        with self.assertRaises(HTTPException) as denied:
            await main.api_get_analysis(
                saved["analysis"]["id"],
                f"Bearer {other_login['token']}",
            )
        self.assertEqual(denied.exception.status_code, 403)

        paths = {route.path for route in main.app.routes}
        self.assertIn("/api/meetings/reservations", paths)
        self.assertIn("/api/meetings/analysis", paths)


if __name__ == "__main__":
    unittest.main()
