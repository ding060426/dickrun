import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main


class SummaryDownloadTests(unittest.TestCase):
    def test_chinese_summary_title_downloads_as_utf8_markdown(self):
        summary = {
            "id": "summary-1",
            "title": "秦朝制度课程摘要",
            "created_by": "user-1",
            "markdown_content": "# 秦朝制度课程摘要\n\n三公九卿与郡县制。\n",
        }

        with (
            patch.object(main, "_require_user", return_value={"id": "user-1", "role": "user"}),
            patch.object(main.summary_store, "get_summary", return_value=summary),
        ):
            response = asyncio.run(
                main.api_download_summary("summary-1", authorization="Bearer test")
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body.decode("utf-8"), summary["markdown_content"])
        disposition = response.headers["content-disposition"]
        self.assertIn('filename="meeting-summary.md"', disposition)
        self.assertIn("filename*=UTF-8''", disposition)

    def test_browser_can_read_download_filename_across_frontend_origin(self):
        summary = {
            "id": "summary-1",
            "title": "课程摘要",
            "created_by": "user-1",
            "markdown_content": "# 课程摘要\n",
        }

        with (
            patch.object(main, "_require_user", return_value={"id": "user-1", "role": "user"}),
            patch.object(main.summary_store, "get_summary", return_value=summary),
        ):
            response = TestClient(main.app).get(
                "/api/record-summaries/summary-1/download",
                headers={
                    "Authorization": "Bearer test",
                    "Origin": "http://127.0.0.1:3000",
                },
            )

        exposed = response.headers.get("access-control-expose-headers", "").lower()
        self.assertIn("content-disposition", exposed)


if __name__ == "__main__":
    unittest.main()
