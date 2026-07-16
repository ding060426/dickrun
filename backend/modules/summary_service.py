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
            max_tokens=4096,
        )
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
            max_tokens=4096,
        )

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
                max_tokens=4096,
            )
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
        max_tokens=8192,
    )

    # Inject per-meeting summaries into result
    result["meeting_summaries"] = result.get("meeting_summaries") or meeting_summaries

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
            markdown = (
                markdown.rstrip()
                + "\n\n## 文字结构图\n\n"
                + "```mermaid\n"
                + diag_mermaid
                + "\n```\n\n"
                + "### Markdown 备用结构\n\n"
                + diag_markdown.rstrip()
                + "\n"
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
