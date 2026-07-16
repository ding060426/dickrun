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
        }
        conn.execute(
            """
            INSERT INTO meeting_records (
                id, meeting_id, title, source_type, source_filename,
                source_mime_type, source_size_bytes, full_text, segments_count,
                duration_sec, speakers_json, metadata_json, created_by,
                created_at, updated_at
            ) VALUES (
                :id, :meeting_id, :title, :source_type, :source_filename,
                :source_mime_type, :source_size_bytes, :full_text, :segments_count,
                :duration_sec, :speakers_json, :metadata_json, :created_by,
                :created_at, :updated_at
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
                updated_at=excluded.updated_at
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
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
    if include_segments:
        return get_record(record_id)
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


def get_record(record_id: str) -> dict | None:
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
        item.update(
            {
                "index": row["segment_index"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "speaker_id": row["speaker_id"],
                "speaker_name": row["speaker_name"],
                "text": row["text"],
                "display_text": row["text"],
                "raw_text": row["raw_text"],
                "audio_wav_base64": _audio_to_base64(row["audio_blob"]),
            }
        )
        result["segments"].append(item)
    return result


def delete_record(record_id: str) -> bool:
    init_db()
    with connect() as conn:
        cursor = conn.execute("DELETE FROM meeting_records WHERE id = ?", (record_id,))
        return cursor.rowcount > 0
