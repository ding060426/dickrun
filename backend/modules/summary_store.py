"""CRUD for LLM-generated record summaries."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .record_store import (
    BACKEND_DIR,
    DB_PATH,
    connect,
    init_db,
    now_iso,
    _json_dump,
    _json_load,
    _number,
)

MAX_RECORDS_PER_SUMMARY = 20


def _summary_to_dict(row: Any) -> dict:
    result = {
        "id": row["id"],
        "title": row["title"],
        "summary_type": row["summary_type"],
        "language": row["language"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "options": _json_load(row["options_json"], {}),
        "result": _json_load(row["result_json"], {}),
        "markdown_content": row["markdown_content"],
        "provider": row["provider"],
        "model_name": row["model_name"],
        "prompt_version": row["prompt_version"],
        "error_message": row["error_message"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }
    optional_columns = {
        "diagram": ("diagram_json", {}),
        "diagram_mermaid": ("diagram_mermaid", ""),
        "diagram_markdown": ("diagram_markdown", ""),
        "llm_settings_snapshot": ("llm_settings_snapshot_json", {}),
        "output_style": ("output_style", "formal"),
        "formula_mode": ("formula_mode", "latex"),
        "diagram_type": ("diagram_type", "auto"),
    }
    row_keys = set(row.keys())
    for public_key, (column, fallback) in optional_columns.items():
        if column not in row_keys:
            result[public_key] = fallback
        elif column.endswith("_json"):
            result[public_key] = _json_load(row[column], fallback)
        else:
            result[public_key] = row[column]
    return result


def create_summary(
    *,
    title: str,
    summary_type: str = "standard",
    language: str = "zh-CN",
    record_ids: list[str],
    options: dict | None = None,
    created_by: str | None = None,
) -> dict:
    init_db()
    summary_id = str(uuid.uuid4())
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO record_summaries (
                id, title, summary_type, language, status, stage,
                progress, options_json, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', '', 0, ?, ?, ?, ?)
            """,
            (
                summary_id,
                (title or "未命名摘要").strip()[:240],
                summary_type,
                language,
                _json_dump(options or {}, {}),
                created_by,
                timestamp,
                timestamp,
            ),
        )
        for idx, record_id in enumerate(record_ids):
            # Snapshot current record title
            rec = conn.execute(
                "SELECT title, updated_at FROM meeting_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            conn.execute(
                """
                INSERT OR IGNORE INTO record_summary_items (
                    summary_id, record_id, sort_order,
                    record_title_snapshot, record_updated_at_snapshot
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    record_id,
                    idx,
                    rec["title"] if rec else "",
                    rec["updated_at"] if rec else "",
                ),
            )
    return get_summary(summary_id)


def get_summary(summary_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM record_summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        if not row:
            return None
        items = conn.execute(
            "SELECT record_id, sort_order, record_title_snapshot, record_updated_at_snapshot FROM record_summary_items WHERE summary_id = ? ORDER BY sort_order",
            (summary_id,),
        ).fetchall()
        # G1c: detect stale records (snapshot != current updated_at) — must be inside the txn
        stale = []
        for item in items:
            if item["record_updated_at_snapshot"]:
                cur = conn.execute(
                    "SELECT updated_at FROM meeting_records WHERE id = ?",
                    (item["record_id"],),
                ).fetchone()
                if cur and cur["updated_at"] != item["record_updated_at_snapshot"]:
                    stale.append(item["record_id"])
    result = _summary_to_dict(row)
    result["record_ids"] = [item["record_id"] for item in items]
    result["record_count"] = len(items)
    result["stale_records"] = stale if stale else []
    return result


def list_summaries(
    *,
    user_id: str | None = None,
    q: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    init_db()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    clauses = []
    params: list[Any] = []
    if user_id:
        clauses.append("created_by = ?")
        params.append(user_id)
    if q:
        clauses.append("title LIKE ?")
        params.append(f"%{q.strip()}%")
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS count FROM record_summaries {where}", params
        ).fetchone()["count"]
        rows = conn.execute(
            f"SELECT * FROM record_summaries {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    items = []
    for row in rows:
        d = _summary_to_dict(row)
        d["markdown_content"] = ""  # trim for list view
        items.append(d)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def update_summary(summary_id: str, patch: dict) -> dict | None:
    init_db()
    allowed = {
        "status", "stage", "progress", "options_json", "result_json",
        "markdown_content", "provider", "model_name", "prompt_version",
        "error_message", "completed_at", "title",
        "diagram_json", "diagram_mermaid", "diagram_markdown",
        "llm_settings_snapshot_json", "output_style", "formula_mode",
        "diagram_type",
    }
    updates = {}
    for k in allowed:
        if k in patch:
            val = patch[k]
            if k in ("options_json", "result_json") and isinstance(val, dict):
                val = json.dumps(val, ensure_ascii=False)
            updates[k] = val
    if not updates:
        return get_summary(summary_id)
    updates["updated_at"] = now_iso()
    set_clauses = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [summary_id]
    with connect() as conn:
        conn.execute(f"UPDATE record_summaries SET {set_clauses} WHERE id = ?", params)
    return get_summary(summary_id)


def delete_summary(summary_id: str) -> bool:
    init_db()
    with connect() as conn:
        cursor = conn.execute("DELETE FROM record_summaries WHERE id = ?", (summary_id,))
        return cursor.rowcount > 0


def get_summary_config(summary_id: str) -> dict | None:
    """Return the original config for retry (record_ids, type, options, etc)."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM record_summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        if not row:
            return None
        items = conn.execute(
            "SELECT record_id FROM record_summary_items WHERE summary_id = ? ORDER BY sort_order",
            (summary_id,),
        ).fetchall()
    return {
        "id": row["id"],
        "title": row["title"],
        "summary_type": row["summary_type"],
        "language": row["language"],
        "options": _json_load(row["options_json"], {}),
        "record_ids": [item["record_id"] for item in items],
        "created_by": row["created_by"],
    }
