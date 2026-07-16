"""System and user prompt templates for meeting summary generation.

Language is parameterised so the prompts can be swapped between zh-CN and en.
"""


def single_summary_system(language: str = "zh-CN") -> str:
    if language.startswith("en"):
        return (
            "You are a professional meeting analyst. Given a meeting transcript, "
            "produce a structured JSON summary covering: overview, topics, decisions, "
            "action items, risks, open questions, and speaker contributions.\n"
            "Only return valid JSON conforming to the provided schema. "
            "Use empty strings/arrays for missing information; never invent facts."
        )
    return (
        "你是一名专业的会议分析师。根据给定的会议转写文本，"
        "生成结构化 JSON 摘要，包含：会议概述、主要议题、关键决策、"
        "行动项、风险、待确认问题、说话人贡献、公式和思维导图结构。\n"
        "只返回符合给定 Schema 的有效 JSON。"
        "缺失信息用空字符串或空数组表示，不要编造事实。\n"
        "每个决策/行动项/风险必须标注 source_record_ids（引用的记录ID）。\n"
        "公式用 LaTeX 语法（$...$ 行内，$$...$$ 块）。图只返回结构化文字节点，"
        "由系统渲染为 Mermaid/Markdown；不要请求或生成图片。图中节点只用简短标签。\n"
        "使用正式、客观、简洁的书面语，避免口语化和空泛修辞。"
    )


def single_summary_user(title: str, date: str, source: str, duration: str, full_text: str, language: str = "zh-CN") -> str:
    header_zh = f"会议标题：{title}\n时间：{date}\n来源：{source}\n时长：{duration}\n\n--- 转写全文 ---\n"
    header_en = f"Meeting Title: {title}\nDate: {date}\nSource: {source}\nDuration: {duration}\n\n--- Full Transcript ---\n"
    header = header_en if language.startswith("en") else header_zh
    return header + full_text


def multi_summary_system(language: str = "zh-CN") -> str:
    if language.startswith("en"):
        return (
            "You are a senior programme manager synthesising insights across multiple "
            "meetings. Given per-meeting structured summaries, produce a comprehensive "
            "JSON report covering: executive summary, per-meeting summaries, common "
            "topics, timeline, decision changes, progress, open action items, resolved "
            "items, new risks, and recommendations.\n"
            "Only return valid JSON conforming to the provided schema. "
            "Cross-reference meeting dates and titles in your analysis."
        )
    return (
        "你是一名资深项目管理者，需要综合多场会议的信息。"
        "根据每场会议的结构化摘要，生成综合 JSON 报告，包含："
        "执行摘要、每场会议概览、共同议题、时间线、决策变化、"
        "项目进展、未完成行动项、已解决问题、新增风险和建议。\n"
        "只返回符合给定 Schema 的有效 JSON。"
        "分析中引用会议日期和标题进行交叉对比。"
    )


def multi_summary_user(meeting_texts: list[str], language: str = "zh-CN") -> str:
    parts = []
    for i, text in enumerate(meeting_texts, 1):
        if language.startswith("en"):
            parts.append(f"=== Meeting {i} ===\n{text}")
        else:
            parts.append(f"=== 会议 {i} ===\n{text}")
    return "\n\n".join(parts)


def chunk_summary_system(language: str = "zh-CN") -> str:
    if language.startswith("en"):
        return (
            "You are a meeting analyst. Summarise this transcript excerpt concisely, "
            "capturing key points, decisions, and action items.\n"
            "Return JSON: { \"topics\": [...], \"decisions\": [...], \"action_items\": [...], \"key_points\": \"...\" }"
        )
    return (
        "你是一名会议分析师。简要概括这段转写摘录，提取要点、决策和行动项。\n"
        "返回 JSON：{ \"topics\": [...], \"decisions\": [...], \"action_items\": [...], \"key_points\": \"...\" }"
    )


def chunk_merge_system(language: str = "zh-CN") -> str:
    if language.startswith("en"):
        return (
            "Combine the following chunk-level summaries into a single coherent "
            "meeting summary JSON matching the standard single-meeting schema.\n"
            "Maintain chronological order. Remove duplicates."
        )
    return (
        "将以下分段摘要合并为一份连贯的会议摘要 JSON，"
        "符合标准单会议 Schema。保持时间顺序，去除重复内容。"
    )
