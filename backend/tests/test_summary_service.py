import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules import record_store, summary_service, summary_store


class _AlternateShapeClient:
    is_configured = True

    def __init__(self):
        self.prompts = []

    async def generate_json(self, **kwargs):
        self.prompts.append(kwargs["user_prompt"])
        if len(self.prompts) == 1:
            return {
                "meeting_summary": "课程讲解了秦朝三公九卿、郡县制及中央集权的形成。",
                "main_topics": [
                    {
                        "topic": "秦朝政治制度",
                        "content": "中央实行三公九卿，地方推行郡县制。",
                    }
                ],
                "key_decisions": [],
                "action_items": [],
                "risks": [],
                "pending_issues": [],
            }
        return {
            "executive_summary": "本次课程围绕秦朝中央与地方政治制度展开。",
            "meeting_overviews": [
                {
                    "meeting_title": "秦朝政治制度课程",
                    "summary": "讲解三公九卿、郡县制和中央集权。",
                }
            ],
            "common_themes": ["秦朝政治制度"],
            "timeline": {"key_events": ["秦朝建立中央集权制度"]},
            "decision_changes": [],
            "project_progress": "完成秦朝制度框架讲解",
            "uncompleted_action_items": [],
            "resolved_issues": [],
            "new_risks": [],
            "recommendations": [],
        }


class SummaryServiceTests(unittest.TestCase):
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

    def test_multi_summary_preserves_transcript_with_alternate_model_fields(self):
        record = record_store.save_record(
            {
                "title": "秦朝政治制度课程",
                "created_by": "user-1",
                "segments": [
                    {
                        "text": "秦朝中央设置三公九卿，地方推行郡县制，形成中央集权。",
                        "start_sec": 0,
                        "end_sec": 8,
                    }
                ],
            }
        )
        second = record_store.save_record(
            {
                "title": "制度复习",
                "created_by": "user-1",
                "segments": [{"text": "复习中央集权制度。", "start_sec": 0, "end_sec": 2}],
            }
        )
        summary = summary_store.create_summary(
            title="课程摘要",
            summary_type="comprehensive",
            record_ids=[record["id"], second["id"]],
            created_by="user-1",
        )
        client = _AlternateShapeClient()

        markdown = asyncio.run(
            summary_service._summarize_multi(
                summary["id"],
                [record["id"]],
                {},
                "zh-CN",
                client,
            )
        )

        self.assertIn("三公九卿", client.prompts[0])
        self.assertIn("课程讲解了秦朝", client.prompts[1])
        self.assertIn("本次课程围绕秦朝", markdown)
        self.assertIn("讲解三公九卿", markdown)
        self.assertIn("秦朝建立中央集权制度", markdown)
        self.assertNotIn("会议摘要均为空", markdown)


if __name__ == "__main__":
    unittest.main()
