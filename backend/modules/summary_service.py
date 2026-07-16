"""Orchestrate LLM-based meeting summary generation.

Flow:
  single record → chunk (if long) → LLM JSON → MD render → save
  multi record  → per-meeting summaries → cross-meeting LLM → MD render → save
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from . import record_store
from . import summary_store
from .llm_client import LLMClient, get_llm_client, LLMNotConfiguredError, LLMResponseError
from .llm_models import model_capabilities
from . import summary_schema
from . import summary_prompts
from . import markdown_renderer
from . import diagram_renderer

logger = logging.getLogger("summary_service")

MAX_CHUNK_CHARS = 8000  # characters per chunk for long transcripts
MAX_TOTAL_CHARS = int(__import__("os").environ.get("DITING_SUMMARY_MAX_TOTAL_CHARS", "500000"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_settings_snapshot(settings: dict) -> dict:
    """Remove credentials before persisting the per-summary configuration."""

    snapshot = {key: value for key, value in settings.items() if key != "api_key"}
    snapshot["has_api_key"] = bool(settings.get("api_key"))
    snapshot["capabilities"] = model_capabilities(str(settings.get("model_name", "")))
    return snapshot


# ── Chunk helpers ────────────────────────────────────────────────

def _chunk_by_segments(segments: list[dict], max_chars: int = MAX_CHUNK_CHARS) -> list[list[dict]]:
    """Split segments into chunks respecting max_chars, never splitting a segment."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for seg in segments:
        text = seg.get("text", "") or seg.get("display_text", "") or ""
        seg_len = len(text)
        if current and current_len + seg_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(seg)
        current_len += seg_len
    if current:
        chunks.append(current)
    return chunks


def _segments_to_text(segments: list[dict], *, include_speaker: bool = True, language: str = "zh-CN") -> str:
    """Convert a list of segment dicts into a readable transcript block."""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker_id") or ""
        text = seg.get("text") or seg.get("display_text") or ""
        start = seg.get("start_sec", 0)
        ts = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
        if include_speaker and speaker:
            lines.append(f"{ts} {speaker}: {text}")
        else:
            lines.append(f"{ts} {text}")
    if language.startswith("en"):
        return "\n".join(lines)
    return "\n".join(lines)


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _first_text(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extractive_overview(full_text: str, limit: int = 360) -> str:
    """Return a content-bearing fallback when a model omits its overview field."""
    lines = []
    for raw_line in str(full_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            line = line.split("]", 1)[1].strip()
        if line:
            lines.append(line)
    text = " ".join(lines)
    if len(text) <= limit:
        return text
    cut = max(text.rfind(mark, 0, limit) for mark in ("。", "！", "？", ".", "!", "?"))
    return text[: cut + 1 if cut >= limit // 2 else limit].rstrip() + "…"


def _normalize_single_result(result: dict, *, fallback_text: str = "") -> dict:
    """Map common OpenAI-compatible model aliases into the canonical summary schema."""
    data = result if isinstance(result, dict) else {}
    overview = _first_text(
        data,
        "overview",
        "meeting_summary",
        "summary",
        "executive_summary",
    ) or _extractive_overview(fallback_text)

    topics = []
    for item in _as_list(data.get("topics") or data.get("main_topics")):
        if isinstance(item, str):
            topic, summary = item.strip(), ""
        elif isinstance(item, dict):
            topic = _first_text(item, "topic", "title", "name", "label")
            summary = _first_text(item, "summary", "content", "description", "key_points")
        else:
            continue
        if topic or summary:
            topics.append({"topic": topic or "内容要点", "summary": summary, "speakers": item.get("speakers", []) if isinstance(item, dict) else []})

    decisions = []
    for item in _as_list(data.get("decisions") or data.get("key_decisions")):
        if isinstance(item, str):
            decisions.append({"content": item})
        elif isinstance(item, dict):
            content = _first_text(item, "content", "decision", "summary", "description")
            if content:
                decisions.append({**item, "content": content})

    actions = []
    for item in _as_list(data.get("action_items") or data.get("tasks") or data.get("todos")):
        if isinstance(item, str):
            actions.append({"task": item})
        elif isinstance(item, dict):
            task = _first_text(item, "task", "content", "action", "description")
            if task:
                actions.append({**item, "task": task})

    risks = []
    for item in _as_list(data.get("risks") or data.get("risk_items")):
        if isinstance(item, str):
            risks.append({"description": item})
        elif isinstance(item, dict):
            description = _first_text(item, "description", "risk", "content", "summary")
            if description:
                risks.append({**item, "description": description})

    contributions = []
    for item in _as_list(data.get("speaker_contributions") or data.get("contributions")):
        if not isinstance(item, dict):
            continue
        speaker = _first_text(item, "speaker", "speaker_name", "name")
        contribution = _first_text(item, "contribution", "content", "summary")
        if speaker and contribution:
            contributions.append({**item, "speaker": speaker, "contribution": contribution})

    normalized = {
        **data,
        "overview": overview,
        "topics": topics,
        "decisions": decisions,
        "action_items": actions,
        "risks": risks,
        "open_questions": [
            str(value).strip()
            for value in _as_list(data.get("open_questions") or data.get("pending_issues"))
            if str(value).strip()
        ],
        "speaker_contributions": contributions,
        "formulas": [item for item in _as_list(data.get("formulas")) if isinstance(item, dict)],
    }
    diagram = data.get("diagram")
    if isinstance(diagram, dict):
        normalized["diagram"] = diagram
    return normalized


def _normalize_multi_result(result: dict) -> dict:
    """Normalize provider-specific multi-summary aliases and container types."""
    data = result if isinstance(result, dict) else {}

    meeting_summaries = []
    raw_meetings = data.get("meeting_summaries") or data.get("meeting_overviews")
    for item in _as_list(raw_meetings):
        if not isinstance(item, dict):
            continue
        meeting_summaries.append({
            **item,
            "record_id": _first_text(item, "record_id", "id"),
            "title": _first_text(item, "title", "meeting_title", "name"),
            "date": _first_text(item, "date", "meeting_date", "created_at"),
            "summary": _first_text(item, "summary", "overview", "meeting_summary"),
            "topics": [str(value).strip() for value in _as_list(item.get("topics") or item.get("key_topics")) if str(value).strip()],
            "decisions": [str(value).strip() for value in _as_list(item.get("decisions") or item.get("decisions_made")) if str(value).strip()],
            "action_items": [str(value).strip() for value in _as_list(item.get("action_items") or item.get("tasks")) if str(value).strip()],
        })

    common_topics = []
    for item in _as_list(data.get("common_topics") or data.get("common_themes")):
        if isinstance(item, str) and item.strip():
            common_topics.append({"topic": item.strip(), "description": "", "mentioned_in": []})
        elif isinstance(item, dict):
            topic = _first_text(item, "topic", "theme", "title", "name")
            description = _first_text(item, "description", "summary", "content")
            if topic or description:
                common_topics.append({**item, "topic": topic or "共同内容", "description": description})

    raw_timeline = data.get("timeline") or []
    if isinstance(raw_timeline, dict):
        raw_timeline = raw_timeline.get("key_events") or raw_timeline.get("events") or []
    timeline = []
    for item in _as_list(raw_timeline):
        if isinstance(item, str) and item.strip():
            timeline.append({"date": "", "event": item.strip()})
        elif isinstance(item, dict):
            event = _first_text(item, "event", "content", "summary", "description")
            if event:
                timeline.append({**item, "date": _first_text(item, "date", "time"), "event": event})

    progress = data.get("progress_changes")
    if progress is None:
        progress = data.get("project_progress")

    open_actions = []
    for item in _as_list(data.get("open_action_items") or data.get("uncompleted_action_items")):
        if isinstance(item, str) and item.strip():
            open_actions.append({"task": item.strip()})
        elif isinstance(item, dict):
            task = _first_text(item, "task", "content", "action", "description")
            if task:
                open_actions.append({**item, "task": task})

    new_risks = []
    for item in _as_list(data.get("new_risks") or data.get("risks")):
        if isinstance(item, str) and item.strip():
            new_risks.append({"description": item.strip()})
        elif isinstance(item, dict):
            description = _first_text(item, "description", "risk", "content")
            if description:
                new_risks.append({**item, "description": description})

    return {
        **data,
        "executive_summary": _first_text(data, "executive_summary", "summary", "overview"),
        "meeting_summaries": meeting_summaries,
        "common_topics": common_topics,
        "timeline": timeline,
        "decision_changes": [item for item in _as_list(data.get("decision_changes")) if isinstance(item, dict)],
        "progress_changes": [str(value).strip() for value in _as_list(progress) if str(value).strip()],
        "open_action_items": open_actions,
        "resolved_items": [str(value).strip() for value in _as_list(data.get("resolved_items") or data.get("resolved_issues")) if str(value).strip()],
        "new_risks": new_risks,
        "recommendations": [str(value).strip() for value in _as_list(data.get("recommendations")) if str(value).strip()],
        "formulas": [item for item in _as_list(data.get("formulas")) if isinstance(item, dict)],
    }


def _merge_meeting_summaries(model_items: list[dict], source_items: list[dict]) -> list[dict]:
    """Keep source IDs and content even when the cross-meeting model omits them."""
    merged = []
    for index, source in enumerate(source_items):
        model = model_items[index] if index < len(model_items) and isinstance(model_items[index], dict) else {}
        merged.append({
            **source,
            **model,
            "record_id": source.get("record_id", ""),
            "title": model.get("title") or source.get("title", ""),
            "date": model.get("date") or source.get("date", ""),
            "summary": model.get("summary") or source.get("summary", ""),
            "topics": model.get("topics") or source.get("topics", []),
            "decisions": model.get("decisions") or source.get("decisions", []),
            "action_items": model.get("action_items") or source.get("action_items", []),
        })
    return merged


# ── Single-meeting summary ───────────────────────────────────────

async def _summarize_single(
    summary_id: str,
    record_ids: list[str],
    options: dict,
    language: str,
    client: LLMClient | None = None,
) -> str:
    """Generate a single-meeting summary. Returns markdown."""
    client = client or get_llm_client()
    if not client.is_configured:
        raise LLMNotConfiguredError("LLM not configured — set DITING_LLM_* environment variables")

    record_id = record_ids[0]
    rec = record_store.get_record(record_id)
    if not rec:
        raise ValueError(f"Record {record_id} not found")

    segments = rec.get("segments") or []
    full_text = rec.get("full_text") or ""

    summary_store.update_summary(summary_id, {"stage": "reading", "progress": 0.05})

    if not full_text.strip():
        full_text = _segments_to_text(segments, language=language)

    meta = {
        "title": rec.get("title", ""),
        "created_at": rec.get("created_at", ""),
        "source_type": rec.get("source_type", ""),
        "speakers_count": len(rec.get("speakers", [])),
        "duration_sec": rec.get("duration_sec", 0),
        "record_id": record_id,
        "record_count": 1,
    }

    # Check if we need chunking
    if len(full_text) <= MAX_CHUNK_CHARS:
        summary_store.update_summary(summary_id, {"stage": "generating", "progress": 0.2})
        result = await client.generate_json(
            system_prompt=summary_prompts.single_summary_system(language),
            user_prompt=summary_prompts.single_summary_user(
                title=meta["title"], date=meta["created_at"], source=meta["source_type"],
                duration=f"{meta['duration_sec']:.1f}s", full_text=full_text, language=language,
            ),
            json_schema=summary_schema.SINGLE_SUMMARY_SCHEMA,
            result_normalizer=lambda value: _normalize_single_result(
                value,
                fallback_text=full_text,
            ),
            max_tokens=4096,
        )
        result = _normalize_single_result(result, fallback_text=full_text)
    else:
        # Chunked processing
        segments_for_chunking = segments if segments else _text_to_segments(full_text)
        chunks = _chunk_by_segments(segments_for_chunking)
        logger.info("Summary %s: chunking %d chars into %d chunks", summary_id, len(full_text), len(chunks))

        summary_store.update_summary(summary_id, {"stage": "chunking", "progress": 0.1})

        # Phase 1: chunk-level summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            progress = 0.1 + 0.4 * (i / len(chunks))
            summary_store.update_summary(summary_id, {
                "stage": f"chunking_{i+1}/{len(chunks)}", "progress": progress,
            })
            chunk_text = _segments_to_text(chunk, language=language)
            chunk_result = await client.generate_json(
                system_prompt=summary_prompts.chunk_summary_system(language),
                user_prompt=chunk_text,
                max_tokens=2048,
            )
            chunk_summaries.append(chunk_result)

        # Phase 2: merge chunk summaries
        summary_store.update_summary(summary_id, {"stage": "merging_chunks", "progress": 0.6})
        merge_text = "\n\n---\n\n".join(
            json.dumps(cs, ensure_ascii=False) for cs in chunk_summaries
        )
        result = await client.generate_json(
            system_prompt=summary_prompts.chunk_merge_system(language),
            user_prompt=merge_text,
            json_schema=summary_schema.SINGLE_SUMMARY_SCHEMA,
            result_normalizer=lambda value: _normalize_single_result(
                value,
                fallback_text=full_text,
            ),
            max_tokens=4096,
        )
        result = _normalize_single_result(result, fallback_text=full_text)

    summary_store.update_summary(summary_id, {"stage": "rendering", "progress": 0.85, "result_json": result})
    markdown = markdown_renderer.render_single_summary(result, meta)
    return markdown


def _text_to_segments(full_text: str) -> list[dict]:
    """Fallback: convert plain text into pseudo-segments for chunking."""
    paragraphs = full_text.split("\n")
    segs = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para:
            segs.append({"text": para, "display_text": para, "start_sec": i * 10, "speaker_name": ""})
    return segs


# ── Multi-meeting synthesis ──────────────────────────────────────

async def _summarize_multi(
    summary_id: str,
    record_ids: list[str],
    options: dict,
    language: str,
    client: LLMClient | None = None,
) -> str:
    """Generate a comprehensive multi-meeting summary. Returns markdown."""
    client = client or get_llm_client()
    if not client.is_configured:
        raise LLMNotConfiguredError("LLM not configured — set DITING_LLM_* environment variables")

    total = len(record_ids)

    # Phase 1: per-meeting lightweight summaries
    meeting_summaries: list[dict] = []
    meeting_metas: list[dict] = []
    for i, rid in enumerate(record_ids):
        progress = 0.05 + 0.45 * (i / total)
        summary_store.update_summary(summary_id, {
            "stage": f"per_meeting_{i+1}/{total}", "progress": progress,
        })
        rec = record_store.get_record(rid)
        if not rec:
            continue
        full_text = rec.get("full_text") or ""
        segments = rec.get("segments") or []
        if not full_text.strip() and segments:
            full_text = _segments_to_text(segments, language=language)

        if len(full_text) > MAX_CHUNK_CHARS:
            # Shorter: truncate to ~4000 chars for intermediate summary
            full_text = full_text[:4000] + "\n...(truncated)"

        meta = {
            "record_id": rid,
            "title": rec.get("title", ""),
            "date": rec.get("created_at", ""),
            "duration_sec": rec.get("duration_sec", 0),
        }
        meeting_metas.append(meta)

        try:
            ms = await client.generate_json(
                system_prompt=summary_prompts.single_summary_system(language),
                user_prompt=summary_prompts.single_summary_user(
                    title=meta["title"], date=meta["date"], source=rec.get("source_type", ""),
                    duration=f"{meta['duration_sec']:.1f}s", full_text=full_text, language=language,
                ),
                json_schema=summary_schema.SINGLE_SUMMARY_SCHEMA,
                result_normalizer=lambda value, transcript=full_text: _normalize_single_result(
                    value,
                    fallback_text=transcript,
                ),
                max_tokens=4096,
            )
            ms = _normalize_single_result(ms, fallback_text=full_text)
            meeting_summaries.append({
                "record_id": rid,
                "title": meta["title"],
                "date": meta["date"],
                "summary": ms.get("overview", ""),
                "topics": [t.get("topic", "") for t in (ms.get("topics") or [])],
                "decisions": [d.get("content", "") for d in (ms.get("decisions") or [])],
                "action_items": [a.get("task", "") for a in (ms.get("action_items") or [])],
            })
        except Exception as exc:
            logger.error("Per-meeting summary failed for %s: %s", rid, exc)
            meeting_summaries.append({
                "record_id": rid,
                "title": meta["title"],
                "date": meta["date"],
                "summary": f"(Summary failed: {exc})",
                "topics": [], "decisions": [], "action_items": [],
            })

    # Phase 2: cross-meeting synthesis
    summary_store.update_summary(summary_id, {"stage": "cross_meeting", "progress": 0.55})
    meeting_texts = [json.dumps(ms, ensure_ascii=False, default=str) for ms in meeting_summaries]

    result = await client.generate_json(
        system_prompt=summary_prompts.multi_summary_system(language),
        user_prompt=summary_prompts.multi_summary_user(meeting_texts, language),
        json_schema=summary_schema.MULTI_SUMMARY_SCHEMA,
        result_normalizer=_normalize_multi_result,
        max_tokens=8192,
    )
    result = _normalize_multi_result(result)

    # Preserve record IDs and source-backed content even if the synthesis model
    # omits them or returns provider-specific aliases.
    result["meeting_summaries"] = _merge_meeting_summaries(
        result.get("meeting_summaries") or [],
        meeting_summaries,
    )
    if not result.get("executive_summary"):
        result["executive_summary"] = " ".join(
            item.get("summary", "")
            for item in result["meeting_summaries"]
            if item.get("summary")
        )

    summary_store.update_summary(summary_id, {"stage": "rendering", "progress": 0.85, "result_json": result})

    meta = {
        "title": summary_store.get_summary(summary_id).get("title", ""),
        "record_count": total,
        "created_at": _now_iso(),
    }
    markdown = markdown_renderer.render_multi_summary(result, meta)
    return markdown


# ── Public entry point ────────────────────────────────────────────

async def generate_summary(summary_id: str) -> None:
    """Main entry point called by the API as an asyncio background task.

    Reads the summary config from the DB, runs the LLM pipeline,
    and saves the result back.
    """
    import json as _json

    config = summary_store.get_summary_config(summary_id)
    if not config:
        logger.error("Summary %s not found", summary_id)
        return

    record_ids = config["record_ids"]
    summary_type = config["summary_type"]
    language = config.get("language", "zh-CN")
    options = config.get("options", {})
    title = config.get("title", "")

    # G3/G4: Load effective LLM settings (user-saved > env vars)
    created_by = config.get("created_by", "")
    effective_settings = {}
    client: LLMClient | None = None
    diagram_enabled = True
    diagram_type = "auto"
    try:
        from . import llm_settings_store
        effective_settings = llm_settings_store.get_effective_settings(created_by, options.get("llm_override"))
        diagram_enabled = effective_settings.get("diagram_enabled", True)
        diagram_type = effective_settings.get("diagram_type", "auto")
    except Exception as e:
        logger.warning("Could not load LLM settings: %s", e)

    try:
        summary_store.update_summary(summary_id, {"status": "processing", "stage": "starting", "progress": 0})
        client = LLMClient.from_settings(effective_settings)
        if not client.is_configured:
            raise LLMNotConfiguredError("LLM base URL and model are not configured")

        if len(record_ids) == 1:
            markdown = await _summarize_single(summary_id, record_ids, options, language, client)
        else:
            markdown = await _summarize_multi(summary_id, record_ids, options, language, client)

        # G4: extract diagram from result_json and render Mermaid
        diag_json = "{}"
        diag_mermaid = ""
        diag_markdown = ""
        if diagram_enabled:
            try:
                summary = summary_store.get_summary(summary_id)
                result = summary.get("result", {}) if summary else {}
                diagram_data = result.get("diagram") if isinstance(result, dict) else None
                if diagram_data:
                    if diagram_type != "auto":
                        diagram_data = dict(diagram_data, type=diagram_type)
                    diag_json = _json.dumps(diagram_data, ensure_ascii=False)
                    diag_mermaid = diagram_renderer.render_mermaid(diagram_data)
                    diag_markdown = diagram_renderer.render_markdown_tree(diagram_data)
            except Exception as e:
                logger.warning("Diagram render failed for %s: %s", summary_id, e)

        if diag_mermaid:
            diagram_content = (
                "```mermaid\n"
                + diag_mermaid
                + "\n```\n\n"
                + "### Markdown 备用结构\n\n"
                + diag_markdown.rstrip()
            )
            markdown = markdown_renderer.append_numbered_section(
                markdown,
                "文字结构图",
                diagram_content,
            )

        summary_store.update_summary(summary_id, {
            "status": "completed",
            "stage": "done",
            "progress": 1.0,
            "markdown_content": markdown,
            "completed_at": _now_iso(),
            "provider": effective_settings.get("provider", ""),
            "model_name": effective_settings.get("model_name", ""),
            "diagram_json": diag_json,
            "diagram_mermaid": diag_mermaid,
            "diagram_markdown": diag_markdown,
            "diagram_type": diagram_type,
            "llm_settings_snapshot_json": _json.dumps(
                _public_settings_snapshot(effective_settings),
                ensure_ascii=False,
                default=str,
            ),
            "output_style": "formal" if effective_settings.get("formal_style", True) else "standard",
            "formula_mode": effective_settings.get("formula_mode", "latex"),
        })
        logger.info("Summary %s completed (%d records, %d chars)", summary_id, len(record_ids), len(markdown))

    except LLMNotConfiguredError:
        summary_store.update_summary(summary_id, {
            "status": "failed",
            "stage": "error",
            "error_message": "LLM not configured. Save Base URL, API Key and model in 大模型设置, or set DITING_LLM_* env vars.",
        })
        logger.warning("Summary %s failed: LLM not configured", summary_id)

    except LLMResponseError as exc:
        summary_store.update_summary(summary_id, {
            "status": "failed",
            "stage": "error",
            "error_message": str(exc)[:1000],
        })
        logger.error("Summary %s LLM error: %s", summary_id, exc)

    except Exception as exc:
        summary_store.update_summary(summary_id, {
            "status": "failed",
            "stage": "error",
            "error_message": f"{type(exc).__name__}: {exc}"[:1000],
        })
        logger.exception("Summary %s unexpected error", summary_id)
    finally:
        if client is not None:
            await client.close()
