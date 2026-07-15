import os
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules import meeting_db
import main
from fastapi import HTTPException


class SharedReservationStoreTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(self.db_path)
        self.original_data_dir = meeting_db.DATA_DIR
        self.original_db_path = meeting_db.DB_PATH
        meeting_db.DATA_DIR = Path(self.db_path).parent
        meeting_db.DB_PATH = Path(self.db_path)
        meeting_db.init_db()

        self.organizer = meeting_db.create_user(
            {"username": "organizer", "password": "secret123"}
        )
        self.colleague = meeting_db.create_user(
            {"username": "colleague", "display_name": "同事甲", "password": "secret123"}
        )
        self.outsider = meeting_db.create_user(
            {"username": "outsider", "password": "secret123"}
        )
        meeting_db.add_friend(self.organizer["id"], self.colleague["id"])

    def tearDown(self):
        meeting_db.DATA_DIR = self.original_data_dir
        meeting_db.DB_PATH = self.original_db_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_participant_sees_shared_reservation_with_people_details(self):
        created = meeting_db.create_reservation(
            {
                "title": "产品周会",
                "organizer_user_id": self.organizer["id"],
                "participant_user_ids": [self.colleague["id"]],
                "start_time": "2026-07-20T09:00:00+08:00",
                "end_time": "2026-07-20T10:00:00+08:00",
            }
        )

        visible = meeting_db.list_reservations(user_id=self.colleague["id"])

        self.assertEqual([meeting["id"] for meeting in visible], [created["id"]])
        self.assertEqual(visible[0]["organizer"]["id"], self.organizer["id"])
        self.assertEqual(
            [participant["id"] for participant in visible[0]["participants"]],
            [self.colleague["id"]],
        )
        self.assertEqual(
            meeting_db.list_reservations(user_id=self.outsider["id"]),
            [],
        )

    def test_reservation_end_must_be_after_start(self):
        with self.assertRaisesRegex(ValueError, "end_time must be after start_time"):
            meeting_db.create_reservation(
                {
                    "title": "错误时间",
                    "organizer_user_id": self.organizer["id"],
                    "start_time": "2026-07-20T10:00:00+08:00",
                    "end_time": "2026-07-20T09:00:00+08:00",
                }
            )

    def test_only_saved_colleagues_can_be_selected(self):
        with self.assertRaisesRegex(ValueError, "active colleagues"):
            meeting_db.create_reservation(
                {
                    "title": "越权邀请",
                    "organizer_user_id": self.organizer["id"],
                    "participant_user_ids": [self.outsider["id"]],
                    "start_time": "2026-07-20T09:00:00+08:00",
                    "end_time": "2026-07-20T10:00:00+08:00",
                }
            )

    def test_legacy_json_participants_are_migrated_without_losing_visibility(self):
        created = meeting_db.create_reservation(
            {
                "title": "旧版预约",
                "organizer_user_id": self.organizer["id"],
                "participant_user_ids": [self.colleague["id"]],
                "start_time": "2026-07-20T09:00:00+08:00",
                "end_time": "2026-07-20T10:00:00+08:00",
            }
        )
        with meeting_db.connect() as conn:
            conn.execute(
                "DELETE FROM meeting_participants WHERE meeting_id = ?",
                (created["id"],),
            )

        self.assertEqual(
            meeting_db.list_reservations(user_id=self.colleague["id"]), []
        )
        meeting_db.init_db()

        migrated = meeting_db.list_reservations(user_id=self.colleague["id"])
        self.assertEqual([reservation["id"] for reservation in migrated], [created["id"]])


class SharedReservationApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        os.unlink(self.db_path)
        self.original_data_dir = meeting_db.DATA_DIR
        self.original_db_path = meeting_db.DB_PATH
        self.original_store = main.db
        meeting_db.DATA_DIR = Path(self.db_path).parent
        meeting_db.DB_PATH = Path(self.db_path)
        meeting_db.init_db()
        main.db = meeting_db

        self.organizer = meeting_db.create_user(
            {"username": "api-organizer", "password": "secret123"}
        )
        self.colleague = meeting_db.create_user(
            {"username": "api-colleague", "password": "secret123"}
        )
        self.outsider = meeting_db.create_user(
            {"username": "api-outsider", "password": "secret123"}
        )
        meeting_db.add_friend(self.organizer["id"], self.colleague["id"])
        self.authorizations = {
            username: f"Bearer {meeting_db.login(username, 'secret123')['token']}"
            for username in ("api-organizer", "api-colleague", "api-outsider")
        }

    def tearDown(self):
        main.db = self.original_store
        meeting_db.DATA_DIR = self.original_data_dir
        meeting_db.DB_PATH = self.original_db_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    async def test_api_lists_only_current_users_reservations(self):
        await main.api_create_reservation(
            {
                "title": "共享会议",
                "participant_user_ids": [self.colleague["id"]],
                "start_time": "2026-07-21T09:00:00+08:00",
                "end_time": "2026-07-21T10:00:00+08:00",
            },
            self.authorizations["api-organizer"],
        )

        participant_view = await main.api_list_reservations(
            None, None, self.authorizations["api-colleague"]
        )
        outsider_view = await main.api_list_reservations(
            None, None, self.authorizations["api-outsider"]
        )

        self.assertEqual(len(participant_view["reservations"]), 1)
        self.assertFalse(participant_view["reservations"][0]["can_manage"])
        organizer_view = await main.api_list_reservations(
            None, None, self.authorizations["api-organizer"]
        )
        self.assertTrue(organizer_view["reservations"][0]["can_manage"])
        self.assertEqual(outsider_view["reservations"], [])

    async def test_only_organizer_can_manage_reservation(self):
        created = await main.api_create_reservation(
            {
                "title": "可管理会议",
                "participant_user_ids": [self.colleague["id"]],
                "start_time": "2026-07-22T09:00:00+08:00",
                "end_time": "2026-07-22T10:00:00+08:00",
            },
            self.authorizations["api-organizer"],
        )
        reservation_id = created["reservation"]["id"]
        self.assertEqual(
            created["reservation"]["organizer_user_id"], self.organizer["id"]
        )

        with self.assertRaises(HTTPException) as denied:
            await main.api_update_reservation(
                reservation_id,
                {"start_time": "2026-07-23T09:00:00+08:00"},
                self.authorizations["api-colleague"],
            )
        self.assertEqual(denied.exception.status_code, 403)

        updated = await main.api_update_reservation(
            reservation_id,
            {
                "start_time": "2026-07-23T09:00:00+08:00",
                "end_time": "2026-07-23T10:30:00+08:00",
                "participant_user_ids": [],
            },
            self.authorizations["api-organizer"],
        )
        self.assertEqual(
            updated["reservation"]["start_time"],
            "2026-07-23T09:00:00+08:00",
        )
        self.assertEqual(updated["reservation"]["participants"], [])


class SupabaseSchemaContractTests(unittest.TestCase):
    def test_normalized_participants_are_synchronized_atomically(self):
        schema_path = BACKEND_DIR / "supabase_init.sql"
        schema = schema_path.read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS meeting_participants", schema)
        self.assertIn("sync_meeting_participants_from_json", schema)
        self.assertIn("AFTER INSERT OR UPDATE OF participant_user_ids", schema)


if __name__ == "__main__":
    unittest.main()
