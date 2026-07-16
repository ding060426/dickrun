import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules import llm_models
from modules import llm_settings_store
from modules import record_store
from modules import summary_service
from modules import summary_store
from modules.llm_client import LLMClient
from modules.summary_service import _public_settings_snapshot


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


class _FakeHttpClient:
    def __init__(self):
        self.requests = []

    async def post(self, path, *, json):
        self.requests.append((path, json))
        return _FakeResponse()


class _FakeSummaryClient:
    is_configured = True

    async def generate_json(self, **_kwargs):
        return {
            "overview": "会议概述",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "risks": [],
            "open_questions": [],
            "speaker_contributions": [],
            "formulas": [],
            "diagram": {
                "type": "flowchart",
                "title": "发布流程",
                "nodes": [
                    {"id": "plan", "label": "计划"},
                    {"id": "ship", "label": "发布", "parent": "plan"},
                ],
            },
        }

    async def close(self):
        return None


class LLMConfigurationTests(unittest.TestCase):
    def test_default_is_current_deepseek_v4_flash(self):
        self.assertEqual(llm_models.DEFAULT_PROVIDER, "deepseek")
        self.assertEqual(llm_models.DEFAULT_MODEL, "deepseek-v4-flash")
        self.assertEqual(
            llm_models.provider_defaults("deepseek")["base_url"],
            "https://api.deepseek.com",
        )

    def test_catalog_includes_v4_pro_and_routes_chat_models_to_text_diagrams(self):
        models = {item["id"]: item for item in llm_models.model_catalog()}
        self.assertIn("deepseek-v4-flash", models)
        self.assertIn("deepseek-v4-pro", models)
        self.assertFalse(models["deepseek-v4-pro"]["image_generation"])
        self.assertEqual(models["deepseek-v4-pro"]["diagram_mode"], "text")

    def test_client_uses_selected_model_and_caps_requested_tokens(self):
        client = LLMClient(
            base_url="https://api.deepseek.com",
            api_key="secret",
            model="deepseek-v4-pro",
            default_max_tokens=2048,
            temperature=0.1,
        )
        fake = _FakeHttpClient()
        client._client = fake

        result = asyncio.run(
            client.generate_json(
                system_prompt="Return JSON.",
                user_prompt="Test",
                max_tokens=4096,
            )
        )

        self.assertTrue(result["ok"])
        path, payload = fake.requests[0]
        self.assertEqual(path, "/chat/completions")
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertNotIn("image", json.dumps(payload))

    def test_summary_snapshot_never_persists_plaintext_api_key(self):
        snapshot = _public_settings_snapshot(
            {
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com",
                "api_key": "top-secret",
                "model_name": "deepseek-v4-pro",
            }
        )

        self.assertNotIn("api_key", snapshot)
        self.assertTrue(snapshot["has_api_key"])
        self.assertEqual(snapshot["model_name"], "deepseek-v4-pro")

    def test_settings_defaults_and_validation_use_dsv4(self):
        defaults = llm_settings_store.default_settings()
        self.assertEqual(defaults["provider"], "deepseek")
        self.assertEqual(defaults["model_name"], "deepseek-v4-flash")

        normalized = llm_settings_store.normalize_settings(
            {
                "provider": "deepseek",
                "model_name": "deepseek-v4-pro",
                "temperature": 99,
                "max_tokens": 1,
                "timeout_sec": 9999,
                "diagram_enabled": False,
            }
        )
        self.assertEqual(normalized["model_name"], "deepseek-v4-pro")
        self.assertEqual(normalized["temperature"], 2.0)
        self.assertEqual(normalized["max_tokens"], 256)
        self.assertEqual(normalized["timeout_sec"], 600)
        self.assertFalse(normalized["diagram_enabled"])

    def test_saved_user_credentials_drive_effective_client_without_leaking_key(self):
        original_db_path = record_store.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as root:
                record_store.DB_PATH = Path(root) / "records.db"
                llm_settings_store.save_settings(
                    "user-1",
                    {
                        "provider": "deepseek",
                        "base_url": "https://api.deepseek.com",
                        "api_key": "saved-secret",
                        "model_name": "deepseek-v4-pro",
                    },
                )
                effective = llm_settings_store.get_effective_settings("user-1")
                public = llm_settings_store.public_effective_settings("user-1")
        finally:
            record_store.DB_PATH = original_db_path

        self.assertEqual(effective["api_key"], "saved-secret")
        self.assertEqual(effective["model_name"], "deepseek-v4-pro")
        self.assertNotIn("api_key", public)
        self.assertTrue(public["has_api_key"])

    def test_text_diagram_is_returned_and_included_in_downloadable_markdown(self):
        original_db_path = record_store.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as root:
                record_store.DB_PATH = Path(root) / "records.db"
                record = record_store.save_record(
                    {
                        "title": "发布会",
                        "created_by": "user-1",
                        "segments": [{"text": "先计划，再发布", "start_sec": 0, "end_sec": 2}],
                    }
                )
                summary = summary_store.create_summary(
                    title="发布摘要",
                    record_ids=[record["id"]],
                    created_by="user-1",
                )
                settings = {
                    **llm_settings_store.DEFAULT_LLM_SETTINGS,
                    "api_key": "test-only",
                }
                fake_client = _FakeSummaryClient()
                with (
                    patch.object(
                        llm_settings_store,
                        "get_effective_settings",
                        return_value=settings,
                    ),
                    patch.object(
                        summary_service.LLMClient,
                        "from_settings",
                        return_value=fake_client,
                    ),
                ):
                    asyncio.run(summary_service.generate_summary(summary["id"]))
                generated = summary_store.get_summary(summary["id"])
        finally:
            record_store.DB_PATH = original_db_path

        self.assertEqual(generated["status"], "completed")
        self.assertIn("```mermaid", generated["markdown_content"])
        self.assertIn("plan --> ship", generated["diagram_mermaid"])
        self.assertEqual(generated["diagram"]["type"], "flowchart")
        self.assertNotIn("test-only", json.dumps(generated, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
