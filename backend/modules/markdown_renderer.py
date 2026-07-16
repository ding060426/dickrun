"""Render validated summary JSON as concise, dynamically numbered Markdown."""

from __future__ import annotations

import re
from typing import Any


def _escape_md(text: Any) -> str:
    """Escape values used inside Markdown table cells."""
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(_escape_md(value) for value in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _chinese_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value == 10:
        return "十"
    if value < 20:
        return "十" + digits[value - 10]
    tens, ones = divmod(value, 10)
    return digits[tens] + "十" + (digits[ones] if ones else "")


def _render_document(title: str, sections: list[tuple[str, str]]) -> str:
    lines = [title.strip(), ""]
    number = 0
    for heading, content in sections:
        content = str(content or "").strip()
        if not content:
            continue
        number += 1
        lines.extend([f"## {_chinese_number(number)}、{heading}", "", content, ""])
    return "\n".join(lines).rstrip() + "\n"


def append_numbered_section(markdown: str, title: str, content: str) -> str:
    """Append a non-empty second-level section using the next visible number."""
    content = str(content or "").strip()
    if not content:
        return markdown
    count = len(re.findall(r"^## [一二三四五六七八九十]+、", markdown, re.MULTILINE))
    return (
        markdown.rstrip()
        + f"\n\n## {_chinese_number(count + 1)}、{title}\n\n"
        + content
        + "\n"
    )


def _formula_markdown(formulas: list[dict]) -> str:
    blocks: list[str] = []
    for item in formulas:
        name = _escape_md(item.get("name"))
        latex = str(item.get("latex") or "").strip()
        if not name and not latex:
            continue
        block = [f"### {name or '公式'}", "", f"$${latex}$$"]
        explanation = str(item.get("explanation") or "").strip()
        if explanation:
            block.extend(["", explanation])
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def render_single_summary(data: dict, meta: dict | None = None) -> str:
    """Render a single meeting and omit every section without real content."""
    meta = meta or {}
    title = _escape_md(meta.get("title") or data.get("title") or "会议摘要")
    sections: list[tuple[str, str]] = []

    info: list[str] = []
    if title:
        info.append(f"- 会议名称：{title}")
    if meta.get("created_at"):
        info.append(f"- 记录时间：{_escape_md(meta['created_at'])}")
    if meta.get("source_type"):
        info.append(f"- 来源：{_escape_md(meta['source_type'])}")
    if meta.get("speakers_count"):
        info.append(f"- 说话人数：{meta['speakers_count']}")
    if meta.get("duration_sec"):
        info.append(f"- 会议时长：{float(meta['duration_sec']):.1f} 秒")
    sections.append(("会议信息", "\n".join(info)))

    sections.append(("会议概述", str(data.get("overview") or "").strip()))

    topic_blocks: list[str] = []
    for index, item in enumerate(data.get("topics") or [], 1):
        topic = _escape_md(item.get("topic"))
        summary = str(item.get("summary") or "").strip()
        if not topic and not summary:
            continue
        block = [f"### {index}. {topic or '议题'}"]
        if summary:
            block.extend(["", summary])
        speakers = [_escape_md(value) for value in item.get("speakers") or [] if _escape_md(value)]
        if speakers:
            block.extend(["", f"*发言人：{', '.join(speakers)}*"])
        topic_blocks.append("\n".join(block))
    sections.append(("主要议题", "\n\n".join(topic_blocks)))

    decision_rows = [
        [item.get("content"), item.get("owner"), item.get("rationale")]
        for item in data.get("decisions") or []
        if _escape_md(item.get("content"))
    ]
    sections.append(("关键决策", _table(decision_rows, ["决策", "负责人", "依据"])))

    action_rows = [
        [item.get("task"), item.get("assignee"), item.get("deadline"), item.get("priority"), item.get("status")]
        for item in data.get("action_items") or []
        if _escape_md(item.get("task"))
    ]
    sections.append(("行动项", _table(action_rows, ["任务", "负责人", "截止时间", "优先级", "状态"])))
    sections.append(("公式与关键技术说明", _formula_markdown(data.get("formulas") or [])))

    risk_lines: list[str] = []
    for item in data.get("risks") or []:
        description = _escape_md(item.get("description"))
        if not description:
            continue
        risk_lines.append(f"- **{description}**")
        if item.get("impact"):
            risk_lines.append(f"  - 影响：{_escape_md(item['impact'])}")
        if item.get("mitigation"):
            risk_lines.append(f"  - 建议措施：{_escape_md(item['mitigation'])}")
    questions = [_escape_md(value) for value in data.get("open_questions") or [] if _escape_md(value)]
    if questions:
        if risk_lines:
            risk_lines.append("")
        risk_lines.append("**待确认事项：**")
        risk_lines.extend(f"- {value}" for value in questions)
    sections.append(("风险与待确认事项", "\n".join(risk_lines)))

    contribution_lines = [
        f"- **{_escape_md(item.get('speaker'))}**：{_escape_md(item.get('contribution'))}"
        for item in data.get("speaker_contributions") or []
        if _escape_md(item.get("speaker")) and _escape_md(item.get("contribution"))
    ]
    sections.append(("说话人贡献", "\n".join(contribution_lines)))

    record_ids = list(meta.get("record_ids") or [])
    if meta.get("record_id") and meta["record_id"] not in record_ids:
        record_ids.insert(0, meta["record_id"])
    sections.append(("来源记录", "\n".join(f"- 记录 ID：{_escape_md(value)}" for value in record_ids if value)))
    return _render_document(f"# 会议摘要：{title}", sections)


def render_multi_summary(data: dict, meta: dict | None = None) -> str:
    """Render a multi-meeting summary with only populated sections."""
    meta = meta or {}
    title = _escape_md(meta.get("title") or "多会议综合摘要")
    sections: list[tuple[str, str]] = []

    scope: list[str] = []
    if meta.get("record_count"):
        scope.append(f"- 覆盖会议数量：{meta['record_count']}")
    if meta.get("created_at"):
        scope.append(f"- 生成时间：{_escape_md(meta['created_at'])}")
    sections.append(("报告范围", "\n".join(scope)))
    sections.append(("执行摘要", str(data.get("executive_summary") or "").strip()))

    meeting_summaries = data.get("meeting_summaries") or []
    meeting_rows = [
        [item.get("date"), item.get("title"), str(item.get("summary") or "")[:200]]
        for item in meeting_summaries
        if _escape_md(item.get("title")) or _escape_md(item.get("summary"))
    ]
    sections.append(("会议概览", _table(meeting_rows, ["时间", "会议", "主要内容"])))

    common_blocks: list[str] = []
    for item in data.get("common_topics") or []:
        topic = _escape_md(item.get("topic"))
        description = str(item.get("description") or "").strip()
        if not topic and not description:
            continue
        block = [f"### {topic or '共同议题'}"]
        if description:
            block.extend(["", description])
        mentioned = [_escape_md(value) for value in item.get("mentioned_in") or [] if _escape_md(value)]
        if mentioned:
            block.extend(["", f"*涉及：{', '.join(mentioned)}*"])
        common_blocks.append("\n".join(block))
    sections.append(("共同议题", "\n\n".join(common_blocks)))

    change_rows = [
        [item.get("decision"), item.get("change"), item.get("original_record"), item.get("latest_record")]
        for item in data.get("decision_changes") or []
        if _escape_md(item.get("decision")) or _escape_md(item.get("change"))
    ]
    sections.append(("关键决策与变化", _table(change_rows, ["决策", "变化情况", "首次会议", "最新会议"])))

    progress_lines = [f"- {_escape_md(value)}" for value in data.get("progress_changes") or [] if _escape_md(value)]
    timeline = data.get("timeline") or []
    if timeline:
        progress_lines.extend(["", "### 时间线", ""])
        progress_lines.extend(
            f"- **{_escape_md(item.get('date'))}** — {_escape_md(item.get('event'))}"
            for item in timeline
            if _escape_md(item.get("event"))
        )
    sections.append(("项目进展变化", "\n".join(progress_lines)))

    action_rows = [
        [item.get("task"), item.get("assignee"), item.get("first_raised"), item.get("latest_status")]
        for item in data.get("open_action_items") or []
        if _escape_md(item.get("task"))
    ]
    sections.append(("未完成行动项", _table(action_rows, ["任务", "负责人", "首次提出", "最新状态"])))
    sections.append(("公式与关键技术说明", _formula_markdown(data.get("formulas") or [])))

    resolved = [f"- {_escape_md(value)}" for value in data.get("resolved_items") or [] if _escape_md(value)]
    sections.append(("已解决事项", "\n".join(resolved)))

    risk_lines: list[str] = []
    for item in data.get("new_risks") or []:
        description = _escape_md(item.get("description"))
        if not description:
            continue
        risk_lines.append(f"- **{description}**")
        if item.get("impact"):
            risk_lines.append(f"  - 影响：{_escape_md(item['impact'])}")
        if item.get("first_seen"):
            risk_lines.append(f"  - 首次出现：{_escape_md(item['first_seen'])}")
    sections.append(("新增风险与待确认问题", "\n".join(risk_lines)))

    recommendations = [f"- {_escape_md(value)}" for value in data.get("recommendations") or [] if _escape_md(value)]
    sections.append(("综合建议", "\n".join(recommendations)))

    source_lines = [
        f"- **{_escape_md(item.get('title'))}** ({_escape_md(item.get('record_id'))})"
        for item in meeting_summaries
        if _escape_md(item.get("record_id"))
    ]
    sections.append(("来源记录", "\n".join(source_lines)))
    return _render_document(f"# 多会议综合摘要：{title}", sections)
