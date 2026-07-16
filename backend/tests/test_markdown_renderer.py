import re
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.markdown_renderer import append_numbered_section, render_multi_summary, render_single_summary


class MarkdownRendererTests(unittest.TestCase):
    def test_single_summary_omits_empty_sections_and_renumbers_remaining_sections(self):
        markdown = render_single_summary(
            {
                "overview": "本次会议确认了交付范围。",
                "topics": [],
                "decisions": [{"content": "周五发布", "owner": "张三", "rationale": "已通过验收"}],
                "action_items": [],
                "formulas": [],
                "risks": [],
                "open_questions": [],
                "speaker_contributions": [],
            },
            {"title": "范围确认会", "record_id": "record-1"},
        )

        headings = re.findall(r"^## ([一二三四五六七八九十]+)、(.+)$", markdown, re.MULTILINE)
        self.assertEqual(
            headings,
            [("一", "会议信息"), ("二", "会议概述"), ("三", "关键决策"), ("四", "来源记录")],
        )
        self.assertNotIn("主要议题", markdown)
        self.assertNotIn("行动项", markdown)
        self.assertNotIn("（无", markdown)

    def test_multi_summary_omits_empty_sections_and_appends_diagram_with_next_number(self):
        markdown = render_multi_summary(
            {
                "executive_summary": "项目按计划推进。",
                "meeting_summaries": [],
                "common_topics": [],
                "decision_changes": [],
                "progress_changes": ["已完成接口联调"],
                "timeline": [],
                "open_action_items": [],
                "formulas": [],
                "resolved_items": [],
                "new_risks": [],
                "recommendations": [],
            },
            {"title": "项目周报", "record_count": 2},
        )
        markdown = append_numbered_section(markdown, "文字结构图", "```mermaid\nflowchart TD\n```")

        headings = re.findall(r"^## ([一二三四五六七八九十]+)、(.+)$", markdown, re.MULTILINE)
        self.assertEqual(
            headings,
            [("一", "报告范围"), ("二", "执行摘要"), ("三", "内容脉络与时间线"), ("四", "文字结构图")],
        )
        self.assertNotIn("（无", markdown)


if __name__ == "__main__":
    unittest.main()
