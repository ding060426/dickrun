"""Render validated summary JSON → Markdown.

The LLM outputs JSON; this module turns it into stable, well-formatted
Markdown that doesn't vary with model or temperature changes.
"""

from __future__ import annotations

from typing import Any


def _escape_md(text: str) -> str:
    """Minimal escaping for table-safe text."""
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def _table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return ""
    cols = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in range(cols)) + " |",
    ]
    for row in rows:
        padded = row + [""] * (cols - len(row))
        lines.append("| " + " | ".join(_escape_md(c) for c in padded) + " |")
    return "\n".join(lines) + "\n"


def render_single_summary(data: dict, meta: dict | None = None) -> str:
    """Render a single-meeting summary JSON to Markdown."""
    meta = meta or {}
    title = meta.get("title", data.get("title", "会议摘要"))
    date = meta.get("created_at", "")
    source = meta.get("source_type", "")
    speakers_count = meta.get("speakers_count", 0)
    duration = meta.get("duration_sec", 0)

    lines = [
        f"# 会议摘要：{_escape_md(title)}",
        "",
        "## 一、会议信息",
        "",
        f"- 记录时间：{date}",
        f"- 来源：{source}",
        f"- 说话人数：{speakers_count}",
        f"- 会议时长：{duration:.1f} 秒" if duration else "- 会议时长：—",
        "",
        "## 二、会议概述",
        "",
        str(data.get("overview") or "（无概述）"),
        "",
        "## 三、主要议题",
        "",
    ]

    topics = data.get("topics") or []
    if topics:
        for i, t in enumerate(topics, 1):
            lines.append(f"### {i}. {_escape_md(t.get('topic', ''))}")
            lines.append(f"{t.get('summary', '')}")
            speakers = t.get("speakers") or []
            if speakers:
                lines.append(f"*发言者：{', '.join(speakers)}*")
            lines.append("")
    else:
        lines.append("（无明确议题）\n")

    lines.extend(["## 四、关键决策", ""])
    decisions = data.get("decisions") or []
    if decisions:
        rows = []
        for d in decisions:
            rows.append([
                _escape_md(d.get("content", "")),
                _escape_md(d.get("owner", "—")),
                _escape_md(d.get("rationale", "")),
            ])
        lines.append(_table(rows, ["决策", "负责人", "依据"]))
    else:
        lines.append("（无明确决策）\n")

    lines.extend(["## 五、行动项", ""])
    actions = data.get("action_items") or []
    if actions:
        rows = []
        for a in actions:
            rows.append([
                _escape_md(a.get("task", "")),
                _escape_md(a.get("assignee", "—")),
                _escape_md(a.get("deadline", "—")),
                _escape_md(a.get("priority", "")),
                _escape_md(a.get("status", "pending")),
            ])
        lines.append(_table(rows, ["任务", "负责人", "截止时间", "优先级", "状态"]))
    else:
        lines.append("（无行动项）\n")

    lines.extend(["## 六、公式与关键技术说明", ""])
    formulas = data.get("formulas") or []
    if formulas:
        for f in formulas:
            name = f.get("name", "")
            latex = f.get("latex", "")
            explanation = f.get("explanation", "")
            lines.append(f"### {_escape_md(name)}")
            lines.append("")
            lines.append(f"$${latex}$$")
            lines.append("")
            if explanation:
                lines.append(f"{explanation}")
                lines.append("")
    else:
        lines.append("（无公式或技术说明）\n")

    lines.extend(["## 七、风险与待确认事项", ""])
    risks = data.get("risks") or []
    if risks:
        for r in risks:
            lines.append(f"- **{_escape_md(r.get('description', ''))}**")
            if r.get("impact"):
                lines.append(f"  - 影响：{_escape_md(r['impact'])}")
            if r.get("mitigation"):
                lines.append(f"  - 建议措施：{_escape_md(r['mitigation'])}")
        lines.append("")
    else:
        lines.append("（无风险记录）\n")

    questions = data.get("open_questions") or []
    if questions:
        lines.append("**待确认事项：**")
        for q in questions:
            lines.append(f"- {_escape_md(q)}")
        lines.append("")

    lines.extend(["## 八、说话人贡献", ""])
    contribs = data.get("speaker_contributions") or []
    if contribs:
        for c in contribs:
            lines.append(f"- **{_escape_md(c.get('speaker', ''))}**：{c.get('contribution', '')}")
        lines.append("")
    else:
        lines.append("（无说话人贡献分析）\n")

    lines.extend(["## 九、来源记录", ""])
    lines.append(f"- 记录 ID：{meta.get('record_id', '—')}")
    if meta.get("record_count", 0) > 1:
        for rid in meta.get("record_ids", []):
            lines.append(f"  - {rid}")

    return "\n".join(lines)


def render_multi_summary(data: dict, meta: dict | None = None) -> str:
    """Render a multi-meeting comprehensive summary JSON to Markdown."""
    meta = meta or {}
    title = meta.get("title", "多会议综合摘要")
    record_count = meta.get("record_count", 0)
    created_at = meta.get("created_at", "")

    lines = [
        f"# 多会议综合摘要：{_escape_md(title)}",
        "",
        "## 一、报告范围",
        "",
        f"- 覆盖会议数量：{record_count}",
        f"- 生成时间：{created_at}",
        "",
        "## 二、执行摘要",
        "",
        str(data.get("executive_summary") or "（无摘要）"),
        "",
        "## 三、会议概览",
        "",
    ]

    meeting_summaries = data.get("meeting_summaries") or []
    if meeting_summaries:
        rows = []
        for ms in meeting_summaries:
            rows.append([
                ms.get("date", ""),
                _escape_md(ms.get("title", "")),
                _escape_md(ms.get("summary", "")[:200]),
            ])
        lines.append(_table(rows, ["时间", "会议", "主要内容"]))
    else:
        lines.append("（无会议概览）\n")

    lines.extend(["## 四、共同议题", ""])
    common = data.get("common_topics") or []
    if common:
        for t in common:
            lines.append(f"### {_escape_md(t.get('topic', ''))}")
            lines.append(f"{t.get('description', '')}")
            mentioned = t.get("mentioned_in") or []
            if mentioned:
                lines.append(f"*涉及：{', '.join(mentioned)}*")
            lines.append("")
    else:
        lines.append("（无共同议题）\n")

    lines.extend(["## 五、关键决策与变化", ""])
    changes = data.get("decision_changes") or []
    if changes:
        rows = []
        for d in changes:
            rows.append([
                _escape_md(d.get("decision", "")),
                _escape_md(d.get("change", "")),
                _escape_md(d.get("original_record", "")),
                _escape_md(d.get("latest_record", "")),
            ])
        lines.append(_table(rows, ["决策", "变化情况", "首次会议", "最新会议"]))
    else:
        lines.append("（无决策变化）\n")

    lines.extend(["## 六、项目进展变化", ""])
    progress = data.get("progress_changes") or []
    if progress:
        for p in progress:
            lines.append(f"- {_escape_md(p)}")
        lines.append("")
    else:
        lines.append("（无进展变化记录）\n")

    # Timeline
    timeline = data.get("timeline") or []
    if timeline:
        lines.extend(["### 时间线", ""])
        for item in timeline:
            lines.append(f"- **{item.get('date', '')}** — {_escape_md(item.get('event', ''))}")
        lines.append("")

    lines.extend(["## 七、未完成行动项", ""])
    open_actions = data.get("open_action_items") or []
    if open_actions:
        rows = []
        for a in open_actions:
            rows.append([
                _escape_md(a.get("task", "")),
                _escape_md(a.get("assignee", "—")),
                _escape_md(a.get("first_raised", "")),
                _escape_md(a.get("latest_status", "")),
            ])
        lines.append(_table(rows, ["任务", "负责人", "首次提出", "最新状态"]))
    else:
        lines.append("（无未完成行动项）\n")

    lines.extend(["## 八、公式与关键技术说明", ""])
    formulas = data.get("formulas") or []
    if formulas:
        for f in formulas:
            name = f.get("name", "")
            latex = f.get("latex", "")
            explanation = f.get("explanation", "")
            lines.append(f"### {_escape_md(name)}")
            lines.append("")
            lines.append(f"$${latex}$$")
            lines.append("")
            if explanation:
                lines.append(f"{explanation}")
                lines.append("")
    else:
        lines.append("（无公式或技术说明）\n")

    lines.extend(["## 九、已解决事项", ""])
    resolved = data.get("resolved_items") or []
    if resolved:
        for r in resolved:
            lines.append(f"- {_escape_md(r)}")
        lines.append("")
    else:
        lines.append("（无已解决事项记录）\n")

    lines.extend(["## 十、新增风险与待确认问题", ""])
    new_risks = data.get("new_risks") or []
    if new_risks:
        for r in new_risks:
            lines.append(f"- **{_escape_md(r.get('description', ''))}**")
            if r.get("impact"):
                lines.append(f"  - 影响：{_escape_md(r['impact'])}")
            if r.get("first_seen"):
                lines.append(f"  - 首次出现：{r['first_seen']}")
        lines.append("")
    else:
        lines.append("（无新增风险）\n")

    lines.extend(["## 十一、综合建议", ""])
    recommendations = data.get("recommendations") or []
    if recommendations:
        for r in recommendations:
            lines.append(f"- {_escape_md(r)}")
        lines.append("")
    else:
        lines.append("（无建议）\n")

    lines.extend(["## 十二、来源记录", ""])
    for ms in meeting_summaries:
        lines.append(f"- **{_escape_md(ms.get('title', ''))}** ({ms.get('record_id', '')})")

    return "\n".join(lines)
