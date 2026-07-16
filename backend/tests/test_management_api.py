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
from modules import record_store


class ManagementApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(self.db_path)
        self.original_data_dir = meeting_db.DATA_DIR
        self.original_db_path = meeting_db.DB_PATH
        self.original_record_db_path = record_store.DB_PATH
        meeting_db.DATA_DIR = Path(self.db_path).parent
        meeting_db.DB_PATH = Path(self.db_path)
        record_store.DB_PATH = Path(f"{self.db_path}.records")
        meeting_db.init_db()
        record_store.init_db()
        main.db = meeting_db

    def tearDown(self):
        meeting_db.DATA_DIR = self.original_data_dir
        meeting_db.DB_PATH = self.original_db_path
        record_store.DB_PATH = self.original_record_db_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        if os.path.exists(f"{self.db_path}.records"):
            os.unlink(f"{self.db_path}.records")

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

    async def test_user_can_update_their_current_profile(self):
        registered = await main.auth_register(
            {"username": "profile-owner", "password": "secret123"}
        )
        login = await main.auth_login(
            {"username": "profile-owner", "password": "secret123"}
        )
        authorization = f"Bearer {login['token']}"
        avatar = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
            "ASsJTYQAAAAASUVORK5CYII="
        )

        updated = await main.auth_update_me(
            {
                "display_name": "资料用户",
                "email": "profile@example.com",
                "phone": "13800138000",
                "avatar_data_url": avatar,
            },
            authorization,
        )
        current = await main.auth_me(authorization)

        self.assertEqual(updated["user"]["id"], registered["user"]["id"])
        self.assertEqual(current["user"]["display_name"], "资料用户")
        self.assertEqual(current["user"]["email"], "profile@example.com")
        self.assertEqual(current["user"]["phone"], "13800138000")
        self.assertEqual(current["user"]["avatar_data_url"], avatar)

    async def test_user_can_save_search_and_download_a_local_meeting_record(self):
        await main.auth_register({"username": "recorder", "password": "secret123"})
        login = await main.auth_login({"username": "recorder", "password": "secret123"})
        authorization = f"Bearer {login['token']}"

        saved = await main.api_create_record(
            {
                "title": "产品周会",
                "source_type": "upload",
                "source_filename": "weekly.wav",
                "speakers": [{"id": "SPEAKER_00", "name": "张三"}],
                "segments": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "speaker_name": "张三",
                        "text": "确认下周发布",
                        "start_sec": 0,
                        "end_sec": 2,
                        "audio_wav_base64": "UklGRg==",
                    }
                ],
            },
            authorization,
        )
        record_id = saved["record"]["id"]

        listed = await main.api_list_records("发布", 50, 0, authorization)
        self.assertEqual(listed["total"], 1)
        loaded = await main.api_get_record(record_id, authorization)
        self.assertEqual(loaded["record"]["segments"][0]["speaker_name"], "张三")
        text_response = await main.api_download_record_text(record_id, authorization)
        self.assertIn("确认下周发布", text_response.body.decode("utf-8"))

        paths = {route.path for route in main.app.routes}
        self.assertIn("/api/records", paths)
        self.assertIn("/api/records/{record_id}/text", paths)

    async def test_user_cannot_promote_or_disable_their_own_account(self):
        registered = await main.auth_register(
            {"username": "ordinary-user", "password": "secret123"}
        )
        login = await main.auth_login(
            {"username": "ordinary-user", "password": "secret123"}
        )
        authorization = f"Bearer {login['token']}"

        with self.assertRaises(HTTPException) as denied:
            await main.api_update_user(
                registered["user"]["id"],
                {"role": "admin", "status": "deleted"},
                authorization,
            )

        self.assertEqual(denied.exception.status_code, 403)
        current = await main.auth_me(authorization)
        self.assertEqual(current["user"]["role"], "user")
        self.assertEqual(current["user"]["status"], "active")

    async def test_profile_rejects_unsafe_avatar_data(self):
        await main.auth_register(
            {"username": "avatar-owner", "password": "secret123"}
        )
        login = await main.auth_login(
            {"username": "avatar-owner", "password": "secret123"}
        )
        authorization = f"Bearer {login['token']}"

        with self.assertRaises(HTTPException) as denied:
            await main.auth_update_me(
                {
                    "display_name": "Avatar Owner",
                    "avatar_data_url": "data:text/html;base64,PHNjcmlwdD4=",
                },
                authorization,
            )

        self.assertEqual(denied.exception.status_code, 400)
        current = await main.auth_me(authorization)
        self.assertEqual(current["user"]["avatar_data_url"], "")


if __name__ == "__main__":
    unittest.main()
