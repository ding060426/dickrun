"""Local SQLite persistence for durable meeting transcription records."""

from __future__ import annotations

import base64
import binascii
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(
    os.environ.get("DITING_RECORDS_DB_PATH", BACKEND_DIR / "data" / "records.db")
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meeting_records (
                id TEXT PRIMARY KEY,
                meeting_id TEXT,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'manual',
                source_filename TEXT NOT NULL DEFAULT '',
                source_mime_type TEXT NOT NULL DEFAULT '',
                source_size_bytes INTEGER NOT NULL DEFAULT 0,
                full_text TEXT NOT NULL DEFAULT '',
                segments_count INTEGER NOT NULL DEFAULT 0,
                duration_sec REAL NOT NULL DEFAULT 0,
                speakers_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meeting_record_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                start_sec REAL NOT NULL DEFAULT 0,
                end_sec REAL NOT NULL DEFAULT 0,
                speaker_id TEXT NOT NULL DEFAULT '',
                speaker_name TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                raw_text TEXT NOT NULL DEFAULT '',
                audio_blob BLOB,
                audio_mime_type TEXT NOT NULL DEFAULT 'audio/wav',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (record_id) REFERENCES meeting_records(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_meeting_records_owner_updated
            ON meeting_records(created_by, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_record_segments_record_index
            ON meeting_record_segments(record_id, segment_index);
            """
        )
        _migrate_db(conn)


def _migrate_db(conn) -> None:
    """幂等迁移：为旧数据库补全新字段。只加不删，可反复执行。"""
    # meeting_records 新增字段
    existing = {row[1] for row in conn.execute("PRAGMA table_info('meeting_records')")}
    record_additions = {
        "status": "TEXT NOT NULL DEFAULT 'completed'",
        "save_status": "TEXT NOT NULL DEFAULT 'saved'",
        "source_audio_path": "TEXT NOT NULL DEFAULT ''",
        "source_audio_sha256": "TEXT NOT NULL DEFAULT ''",
        "source_sample_rate": "INTEGER NOT NULL DEFAULT 0",
        "source_channels": "INTEGER NOT NULL DEFAULT 0",
        "source_duration_sec": "REAL NOT NULL DEFAULT 0",
        "processing_stage": "TEXT NOT NULL DEFAULT ''",
        "error_message": "TEXT NOT NULL DEFAULT ''",
        "completed_at": "TEXT NOT NULL DEFAULT ''",
    }
    for col, spec in record_additions.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE meeting_records ADD COLUMN {col} {spec}")

    # meeting_record_segments 新增字段
    existing_seg = {row[1] for row in conn.execute("PRAGMA table_info('meeting_record_segments')")}
    seg_additions = {
        "segment_uuid": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
        "text_revision": "INTEGER NOT NULL DEFAULT 1",
    }
    for col, spec in seg_additions.items():
        if col not in existing_seg:
            conn.execute(f"ALTER TABLE meeting_record_segments ADD COLUMN {col} {spec}")

    # 为已有 segment 补填 UUID（仅处理空 UUID 的行）
    conn.execute(
        """
        UPDATE meeting_record_segments
        SET segment_uuid = lower(hex(randomblob(16)))
        WHERE segment_uuid = ''
        """
    )

    # Summary tables (S2)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS record_summaries (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary_type TEXT NOT NULL DEFAULT 'standard',
            language TEXT NOT NULL DEFAULT 'zh-CN',
            status TEXT NOT NULL DEFAULT 'pending',
            stage TEXT NOT NULL DEFAULT '',
            progress REAL NOT NULL DEFAULT 0,
            options_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            markdown_content TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            model_name TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            created_by TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS record_summary_items (
            summary_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            record_title_snapshot TEXT NOT NULL DEFAULT '',
            record_updated_at_snapshot TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (summary_id, record_id),
            FOREIGN KEY (summary_id) REFERENCES record_summaries(id) ON DELETE CASCADE,
            FOREIGN KEY (record_id) REFERENCES meeting_records(id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS user_llm_settings (
            user_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL DEFAULT '',
            base_url TEXT NOT NULL DEFAULT '',
            api_key_encrypted TEXT NOT NULL DEFAULT '',
            model_name TEXT NOT NULL DEFAULT '',
            temperature REAL NOT NULL DEFAULT 0.2,
            max_tokens INTEGER NOT NULL DEFAULT 8192,
            timeout_sec INTEGER NOT NULL DEFAULT 180,
            diagram_enabled INTEGER NOT NULL DEFAULT 1,
            diagram_type TEXT NOT NULL DEFAULT 'auto',
            output_language TEXT NOT NULL DEFAULT 'zh-CN',
            formal_style INTEGER NOT NULL DEFAULT 1,
            formula_mode TEXT NOT NULL DEFAULT 'latex',
            updated_at TEXT NOT NULL
        );
        """
    )

    # G3a: extend record_summaries with diagram / formula fields
    for col, spec in [
        ("diagram_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("diagram_mermaid", "TEXT NOT NULL DEFAULT ''"),
        ("diagram_markdown", "TEXT NOT NULL DEFAULT ''"),
        ("llm_settings_snapshot_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("output_style", "TEXT NOT NULL DEFAULT 'formal'"),
        ("formula_mode", "TEXT NOT NULL DEFAULT 'latex'"),
        ("diagram_type", "TEXT NOT NULL DEFAULT 'auto'"),
    ]:
        existing = {r[1] for r in conn.execute("PRAGMA table_info('record_summaries')")}
        if col not in existing:
            conn.execute(f"ALTER TABLE record_summaries ADD COLUMN {col} {spec}")


def _json_dump(value: Any, fallback: Any) -> str:
    return json.dumps(value if value is not None else fallback, ensure_ascii=False)


def _json_load(value: str | None, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return fallback


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _audio_from_base64(value: Any) -> tuple[bytes | None, str]:
    if not value or not isinstance(value, str):
        return None, "audio/wav"
    encoded = value
    mime_type = "audio/wav"
    if value.startswith("data:") and "," in value:
        header, encoded = value.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or mime_type
    try:
        return base64.b64decode(encoded, validate=True), mime_type
    except (binascii.Error, ValueError):
        return None, mime_type


def _audio_to_base64(value: bytes | None) -> str | None:
    return base64.b64encode(value).decode("ascii") if value else None


def build_full_text(segments: list[dict]) -> str:
    lines = []
    for segment in segments:
        text = str(
            segment.get("display_text")
            or segment.get("text")
            or segment.get("raw_text")
            or ""
        ).strip()
        if not text:
            continue
        speaker = str(
            segment.get("speaker_name")
            or segment.get("speaker")
            or segment.get("speaker_id")
            or "未区分说话人"
        ).strip()
        lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


def _segment_metadata(segment: dict) -> dict:
    excluded = {
        "audio_wav_base64",
        "index",
        "segment_index",
        "start",
        "start_sec",
        "end",
        "end_sec",
        "speaker",
        "speaker_id",
        "speaker_name",
        "text",
        "display_text",
        "raw_text",
    }
    return {key: value for key, value in segment.items() if key not in excluded}


# ── Incremental save helpers (P0) ────────────────────────────────

def create_draft(title: str = "", *, created_by: str | None = None, **extra) -> dict:
    """立即创建一条 draft 记录，不包含任何分段。返回 summary dict。"""
    init_db()
    record_id = str(extra.pop("id", uuid.uuid4()))
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO meeting_records (
                id, meeting_id, title, source_type, source_filename,
                source_mime_type, source_size_bytes, full_text, segments_count,
                duration_sec, speakers_json, metadata_json, status, save_status,
                created_by, created_at, updated_at
            ) VALUES (
                :id, NULL, :title, :source_type, :source_filename,
                :source_mime_type, :source_size_bytes, '', 0,
                0.0, '[]', '{}', 'draft', 'unsaved',
                :created_by, :created_at, :updated_at
            )
            """,
            {
                "id": record_id,
                "title": (title or "未命名会议记录").strip()[:240],
                "source_type": extra.get("source_type", "manual"),
                "source_filename": extra.get("source_filename", ""),
                "source_mime_type": extra.get("source_mime_type", ""),
                "source_size_bytes": int(_number(extra.get("source_size_bytes", 0))),
                "created_by": created_by,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        row = conn.execute("SELECT * FROM meeting_records WHERE id = ?", (record_id,)).fetchone()
    return _record_to_summary(row)


def append_segment(record_id: str, segment: dict) -> dict:
    """向已有记录追加一个最终分段（含音频 BLOB），更新父记录统计。"""
    init_db()
    seg_uuid = str(segment.get("segment_uuid") or uuid.uuid4())
    timestamp = now_iso()
    audio_blob, audio_mime = _audio_from_base64(segment.get("audio_wav_base64"))
    with connect() as conn:
        record = conn.execute(
            "SELECT id FROM meeting_records WHERE id = ?", (record_id,)
        ).fetchone()
        if not record:
            raise LookupError(f"record {record_id} not found")

        # 计算下一个 segment_index
        max_idx = conn.execute(
            "SELECT COALESCE(MAX(segment_index), 0) AS mx FROM meeting_record_segments WHERE record_id = ?",
            (record_id,),
        ).fetchone()["mx"]

        conn.execute(
            """
            INSERT INTO meeting_record_segments (
                record_id, segment_index, start_sec, end_sec, speaker_id,
                speaker_name, text, raw_text, audio_blob, audio_mime_type,
                metadata_json, segment_uuid, created_at, updated_at, text_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                record_id,
                int(_number(segment.get("index", segment.get("segment_index", max_idx + 1)), max_idx + 1)),
                _number(segment.get("start_sec", segment.get("start", 0))),
                _number(segment.get("end_sec", segment.get("end", 0))),
                segment.get("speaker_id") or "",
                segment.get("speaker_name") or segment.get("speaker") or "",
                segment.get("display_text") or segment.get("text") or "",
                segment.get("raw_text") or "",
                audio_blob,
                audio_mime,
                _json_dump(_segment_metadata(segment), {}),
                seg_uuid,
                timestamp,
                timestamp,
            ),
        )

        # 更新父记录统计
        stats = conn.execute(
            """
            SELECT COUNT(*) AS cnt, COALESCE(MAX(end_sec), 0) AS dur
            FROM meeting_record_segments WHERE record_id = ?
            """,
            (record_id,),
        ).fetchone()
        conn.execute(
            "UPDATE meeting_records SET segments_count = ?, duration_sec = ?, updated_at = ? WHERE id = ?",
            (stats["cnt"], stats["dur"], timestamp, record_id),
        )

    return {
        "segment_uuid": seg_uuid,
        "segment_index": int(_number(segment.get("index", segment.get("segment_index", max_idx + 1)), max_idx + 1)),
        "created_at": timestamp,
    }


def update_segment(record_id: str, segment_uuid: str, patch: dict) -> dict | None:
    """更新单个分段的文本/说话人/元数据。不传音频。返回更新后的 segment 摘要。"""
    init_db()
    timestamp = now_iso()
    with connect() as conn:
        existing = conn.execute(
            "SELECT id, text_revision FROM meeting_record_segments WHERE record_id = ? AND segment_uuid = ?",
            (record_id, segment_uuid),
        ).fetchone()
        if not existing:
            return None

        new_revision = existing["text_revision"] + 1
        updates = {}
        for key in ("text", "display_text", "raw_text", "speaker_id", "speaker_name"):
            if key in patch:
                updates[key] = patch[key]
        if "text" in patch and "display_text" not in patch:
            updates["display_text"] = patch["text"]

        if updates:
            set_clauses = ", ".join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [new_revision, timestamp, record_id, segment_uuid]
            conn.execute(
                f"UPDATE meeting_record_segments SET {set_clauses}, text_revision = ?, updated_at = ? WHERE record_id = ? AND segment_uuid = ?",
                params,
            )

        conn.execute(
            "UPDATE meeting_records SET updated_at = ? WHERE id = ?",
            (timestamp, record_id),
        )

    return {"segment_uuid": segment_uuid, "text_revision": new_revision, "updated_at": timestamp}


def delete_segment(record_id: str, segment_uuid: str) -> bool:
    """删除单个分段，更新父记录统计。"""
    init_db()
    with connect() as conn:
        cursor = conn.execute(
            "DELETE FROM meeting_record_segments WHERE record_id = ? AND segment_uuid = ?",
            (record_id, segment_uuid),
        )
        if cursor.rowcount == 0:
            return False
        stats = conn.execute(
            """
            SELECT COUNT(*) AS cnt, COALESCE(MAX(end_sec), 0) AS dur
            FROM meeting_record_segments WHERE record_id = ?
            """,
            (record_id,),
        ).fetchone()
        conn.execute(
            "UPDATE meeting_records SET segments_count = ?, duration_sec = ?, updated_at = ? WHERE id = ?",
            (stats["cnt"], stats["dur"], now_iso(), record_id),
        )
    return True


def patch_record(record_id: str, patch: dict) -> dict | None:
    """只更新 parent 记录元数据字段（title、source_* 等），不动 segments。"""
    init_db()
    allowed = {
        "title", "source_type", "source_filename", "source_mime_type",
        "source_size_bytes", "meeting_id", "status", "save_status",
        "source_audio_path", "source_audio_sha256", "source_sample_rate",
        "source_channels", "source_duration_sec", "processing_stage",
        "error_message", "completed_at",
    }
    updates = {k: patch[k] for k in allowed if k in patch}
    if not updates:
        with connect() as conn:
            row = conn.execute("SELECT * FROM meeting_records WHERE id = ?", (record_id,)).fetchone()
        return _record_to_summary(row) if row else None

    updates["updated_at"] = now_iso()
    set_clauses = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [record_id]
    with connect() as conn:
        conn.execute(f"UPDATE meeting_records SET {set_clauses} WHERE id = ?", params)
        row = conn.execute("SELECT * FROM meeting_records WHERE id = ?", (record_id,)).fetchone()
    return _record_to_summary(row) if row else None


def finalize_record(record_id: str) -> dict | None:
    """标记记录为 completed：重建 full_text、更新统计、写入 completed_at。"""
    init_db()
    with connect() as conn:
        record = conn.execute("SELECT * FROM meeting_records WHERE id = ?", (record_id,)).fetchone()
        if not record:
            return None
        rows = conn.execute(
            "SELECT * FROM meeting_record_segments WHERE record_id = ? ORDER BY segment_index, id",
            (record_id,),
        ).fetchall()
    segments = []
    for row in rows:
        item = _json_load(row["metadata_json"], {})
        item.update({
            "display_text": row["text"],
            "text": row["text"],
            "raw_text": row["raw_text"],
            "speaker_id": row["speaker_id"],
            "speaker_name": row["speaker_name"],
        })
        segments.append(item)
    full_text = build_full_text(segments) if segments else (record["full_text"] or "")
    duration = max(
        [_number(item.get("end_sec", item.get("end", 0))) for item in segments] or [0]
    )
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            UPDATE meeting_records SET
                full_text = ?, segments_count = ?, duration_sec = ?,
                status = 'completed', save_status = 'saved',
                completed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (full_text, len(segments), duration, timestamp, timestamp, record_id),
        )
    return get_record(record_id)


def _record_to_summary(row: sqlite3.Row) -> dict:
    full_text = row["full_text"] or ""
    return {
        "id": row["id"],
        "meeting_id": row["meeting_id"],
        "title": row["title"],
        "source_type": row["source_type"],
        "source_filename": row["source_filename"],
        "source_mime_type": row["source_mime_type"],
        "source_size_bytes": row["source_size_bytes"],
        "full_text_preview": full_text[:500],
        "segments_count": row["segments_count"],
        "duration_sec": row["duration_sec"],
        "speakers": _json_load(row["speakers_json"], []),
        "metadata": _json_load(row["metadata_json"], {}),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"] if "status" in row.keys() else "completed",
        "save_status": row["save_status"] if "save_status" in row.keys() else "saved",
        "source_audio_path": row["source_audio_path"] if "source_audio_path" in row.keys() else "",
        "source_audio_sha256": row["source_audio_sha256"] if "source_audio_sha256" in row.keys() else "",
        "source_sample_rate": row["source_sample_rate"] if "source_sample_rate" in row.keys() else 0,
        "source_channels": row["source_channels"] if "source_channels" in row.keys() else 0,
        "source_duration_sec": row["source_duration_sec"] if "source_duration_sec" in row.keys() else 0,
        "processing_stage": row["processing_stage"] if "processing_stage" in row.keys() else "",
        "error_message": row["error_message"] if "error_message" in row.keys() else "",
        "completed_at": row["completed_at"] if "completed_at" in row.keys() else "",
    }


def save_record(
    data: dict,
    record_id: str | None = None,
    *,
    include_segments: bool = True,
) -> dict:
    init_db()
    segments = [item for item in (data.get("segments") or []) if isinstance(item, dict)]
    record_id = str(record_id or data.get("id") or uuid.uuid4())
    title = str(data.get("title") or "未命名会议记录").strip()[:240]
    full_text = str(data.get("full_text") or build_full_text(segments)).strip()
    duration = max(
        [_number(item.get("end_sec", item.get("end", 0))) for item in segments] or [0]
    )
    duration = _number(data.get("duration_sec"), duration) or duration
    timestamp = now_iso()

    with connect() as conn:
        existing = conn.execute(
            "SELECT created_at, created_by FROM meeting_records WHERE id = ?",
            (record_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else timestamp
        created_by = data.get("created_by") or (existing["created_by"] if existing else None)
        values = {
            "id": record_id,
            "meeting_id": data.get("meeting_id"),
            "title": title,
            "source_type": data.get("source_type") or "manual",
            "source_filename": data.get("source_filename") or "",
            "source_mime_type": data.get("source_mime_type") or "",
            "source_size_bytes": int(_number(data.get("source_size_bytes"))),
            "full_text": full_text,
            "segments_count": len(segments),
            "duration_sec": duration,
            "speakers_json": _json_dump(data.get("speakers"), []),
            "metadata_json": _json_dump(data.get("metadata"), {}),
            "created_by": created_by,
            "created_at": created_at,
            "updated_at": timestamp,
            "status": data.get("status") or (existing["status"] if existing and "status" in existing.keys() else "completed"),
            "save_status": data.get("save_status") or "saved",
            "source_audio_path": data.get("source_audio_path") or "",
            "source_audio_sha256": data.get("source_audio_sha256") or "",
            "source_sample_rate": int(_number(data.get("source_sample_rate"))),
            "source_channels": int(_number(data.get("source_channels"))),
            "source_duration_sec": _number(data.get("source_duration_sec")),
            "processing_stage": data.get("processing_stage") or "",
            "error_message": data.get("error_message") or "",
            "completed_at": data.get("completed_at") or "",
        }
        conn.execute(
            """
            INSERT INTO meeting_records (
                id, meeting_id, title, source_type, source_filename,
                source_mime_type, source_size_bytes, full_text, segments_count,
                duration_sec, speakers_json, metadata_json, created_by,
                created_at, updated_at, status, save_status,
                source_audio_path, source_audio_sha256, source_sample_rate,
                source_channels, source_duration_sec, processing_stage,
                error_message, completed_at
            ) VALUES (
                :id, :meeting_id, :title, :source_type, :source_filename,
                :source_mime_type, :source_size_bytes, :full_text, :segments_count,
                :duration_sec, :speakers_json, :metadata_json, :created_by,
                :created_at, :updated_at, :status, :save_status,
                :source_audio_path, :source_audio_sha256, :source_sample_rate,
                :source_channels, :source_duration_sec, :processing_stage,
                :error_message, :completed_at
            )
            ON CONFLICT(id) DO UPDATE SET
                meeting_id=excluded.meeting_id,
                title=excluded.title,
                source_type=excluded.source_type,
                source_filename=excluded.source_filename,
                source_mime_type=excluded.source_mime_type,
                source_size_bytes=excluded.source_size_bytes,
                full_text=excluded.full_text,
                segments_count=excluded.segments_count,
                duration_sec=excluded.duration_sec,
                speakers_json=excluded.speakers_json,
                metadata_json=excluded.metadata_json,
                created_by=excluded.created_by,
                updated_at=excluded.updated_at,
                status=excluded.status,
                save_status=excluded.save_status,
                source_audio_path=excluded.source_audio_path,
                source_audio_sha256=excluded.source_audio_sha256,
                source_sample_rate=excluded.source_sample_rate,
                source_channels=excluded.source_channels,
                source_duration_sec=excluded.source_duration_sec,
                processing_stage=excluded.processing_stage,
                error_message=excluded.error_message,
                completed_at=excluded.completed_at
            """,
            values,
        )
        conn.execute("DELETE FROM meeting_record_segments WHERE record_id = ?", (record_id,))
        for position, segment in enumerate(segments, start=1):
            audio_blob, audio_mime_type = _audio_from_base64(
                segment.get("audio_wav_base64")
            )
            conn.execute(
                """
                INSERT INTO meeting_record_segments (
                    record_id, segment_index, start_sec, end_sec, speaker_id,
                    speaker_name, text, raw_text, audio_blob, audio_mime_type,
                    metadata_json, segment_uuid, created_at, updated_at, text_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    int(_number(segment.get("index", segment.get("segment_index", position)), position)),
                    _number(segment.get("start_sec", segment.get("start", 0))),
                    _number(segment.get("end_sec", segment.get("end", 0))),
                    segment.get("speaker_id") or "",
                    segment.get("speaker_name") or segment.get("speaker") or "",
                    segment.get("display_text") or segment.get("text") or "",
                    segment.get("raw_text") or "",
                    audio_blob,
                    audio_mime_type,
                    _json_dump(_segment_metadata(segment), {}),
                    str(segment.get("segment_uuid") or uuid.uuid4()),
                    segment.get("created_at") or timestamp,
                    segment.get("updated_at") or timestamp,
                    int(_number(segment.get("text_revision", 1), 1)),
                ),
            )
    if include_segments:
        # The save API historically returns the complete payload it accepted,
        # including per-segment audio.  Normal GET/list calls can still omit
        # audio and fetch it on demand through the dedicated endpoint.
        return get_record(record_id, include_audio=True)
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM meeting_records WHERE id = ?", (record_id,)
        ).fetchone()
    return _record_to_summary(row)


def list_records(
    *,
    user_id: str | None = None,
    q: str | None = None,
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
        clauses.append("(title LIKE ? OR source_filename LIKE ? OR full_text LIKE ?)")
        like = f"%{q.strip()}%"
        params.extend([like, like, like])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS count FROM meeting_records {where}", params
        ).fetchone()["count"]
        rows = conn.execute(
            f"""
            SELECT * FROM meeting_records {where}
            ORDER BY updated_at DESC LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    return {
        "items": [_record_to_summary(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "q": q or "",
    }


def get_record(record_id: str, *, include_audio: bool = False) -> dict | None:
    init_db()
    with connect() as conn:
        record = conn.execute(
            "SELECT * FROM meeting_records WHERE id = ?", (record_id,)
        ).fetchone()
        if not record:
            return None
        rows = conn.execute(
            """
            SELECT * FROM meeting_record_segments
            WHERE record_id = ? ORDER BY segment_index, id
            """,
            (record_id,),
        ).fetchall()
    result = _record_to_summary(record)
    result["full_text"] = record["full_text"] or ""
    result["segments"] = []
    for row in rows:
        item = _json_load(row["metadata_json"], {})
        segment_data = {
            "index": row["segment_index"],
            "segment_uuid": row["segment_uuid"] if "segment_uuid" in row.keys() else "",
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "speaker_id": row["speaker_id"],
            "speaker_name": row["speaker_name"],
            "text": row["text"],
            "display_text": row["text"],
            "raw_text": row["raw_text"],
            "text_revision": row["text_revision"] if "text_revision" in row.keys() else 1,
            "created_at": row["created_at"] if "created_at" in row.keys() else "",
            "updated_at": row["updated_at"] if "updated_at" in row.keys() else "",
        }
        if include_audio:
            segment_data["audio_wav_base64"] = _audio_to_base64(row["audio_blob"])
        item.update(segment_data)
        result["segments"].append(item)
    return result


def get_segment_audio(record_id: str, segment_uuid: str) -> tuple[bytes | None, str]:
    """Return (audio_blob, mime_type) for a single segment — for on-demand binary endpoint."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            """SELECT audio_blob, audio_mime_type FROM meeting_record_segments
               WHERE record_id = ? AND segment_uuid = ?""",
            (record_id, segment_uuid),
        ).fetchone()
        if not row:
            return None, "audio/wav"
        return row["audio_blob"], row["audio_mime_type"]


def delete_record(record_id: str) -> bool:
    init_db()
    # Clean up associated media files
    try:
        from . import record_media
        record_media.delete_record_media(record_id)
    except Exception:
        pass  # media cleanup is best-effort; DB delete proceeds regardless
    with connect() as conn:
        cursor = conn.execute("DELETE FROM meeting_records WHERE id = ?", (record_id,))
        return cursor.rowcount > 0
