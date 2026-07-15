"""
谛听 DiTing - 本地会议记忆数据库
==============================================================================
使用 Python 标准库 sqlite3 保存上传音视频识别后的文本、片段、摘要、热词和说话人分布。

设计原则：
- 不引入 ORM / 第三方依赖
- 每次操作独立连接，适配 FastAPI 后台线程
- 不保存 audio_wav_base64，避免数据库体积暴涨
- 数据库失败不应影响 ASR 主流程
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional


BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "storage", "diting.db")


def get_db_path() -> str:
    """返回本地数据库路径，可通过 DITING_DB_PATH 覆盖。"""
    return os.environ.get("DITING_DB_PATH", DEFAULT_DB_PATH)


def _json_dumps(value) -> str:
    return json.dumps(value if value is not None else None, ensure_ascii=False)


def _json_loads(value, default=None):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """为已存在的旧表补齐新增字段，避免升级后查询失败。"""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def init_db() -> None:
    """初始化本地记忆数据库，幂等，并兼容旧版本表结构。"""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT,
                engine TEXT,
                created_at TEXT,
                updated_at TEXT,
                segments_count INTEGER DEFAULT 0,
                duration_sec REAL DEFAULT 0,
                full_text TEXT,
                summary_json TEXT,
                hotwords_json TEXT,
                speaker_stats_json TEXT,
                asr_optimizer_json TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                segment_index INTEGER,
                speaker_id TEXT,
                start_sec REAL,
                end_sec REAL,
                text TEXT,
                raw_text TEXT,
                asr_confidence REAL,
                snr_db REAL,
                quality_score REAL,
                quality_label TEXT,
                terms_json TEXT,
                data_points_json TEXT,
                corrections_json TEXT,
                logic_flags_json TEXT,
                uncertain_spans_json TEXT,
                uncertainty_json TEXT,
                created_at TEXT,
                FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_columns(conn, "meetings", {
            "filename": "TEXT",
            "status": "TEXT",
            "engine": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
            "segments_count": "INTEGER DEFAULT 0",
            "duration_sec": "REAL DEFAULT 0",
            "full_text": "TEXT",
            "summary_json": "TEXT",
            "hotwords_json": "TEXT",
            "speaker_stats_json": "TEXT",
            "asr_optimizer_json": "TEXT",
            "metadata_json": "TEXT",
        })
        _ensure_columns(conn, "segments", {
            "meeting_id": "TEXT",
            "segment_index": "INTEGER",
            "speaker_id": "TEXT",
            "start_sec": "REAL",
            "end_sec": "REAL",
            "text": "TEXT",
            "raw_text": "TEXT",
            "asr_confidence": "REAL",
            "snr_db": "REAL",
            "quality_score": "REAL",
            "quality_label": "TEXT",
            "terms_json": "TEXT",
            "data_points_json": "TEXT",
            "corrections_json": "TEXT",
            "logic_flags_json": "TEXT",
            "uncertain_spans_json": "TEXT",
            "uncertainty_json": "TEXT",
            "created_at": "TEXT",
        })
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_meeting_id ON segments(meeting_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_text ON segments(text)")


def strip_audio_payload(segments: list[dict]) -> list[dict]:
    """移除前端波形播放用的大体积音频字段。"""
    clean = []
    for seg in segments or []:
        if isinstance(seg, dict):
            item = dict(seg)
            item.pop("audio_wav_base64", None)
            clean.append(item)
    return clean


def build_full_text(segments: list[dict]) -> str:
    """按说话人拼接完整转写文本。"""
    lines = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker_id") or seg.get("speaker") or "unknown"
        text = seg.get("text") or seg.get("display_text") or seg.get("raw_text") or ""
        text = str(text).strip()
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


def _duration_from_segments(segments: list[dict]) -> float:
    end_values = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        end = seg.get("end_sec", seg.get("end", 0))
        try:
            end_values.append(float(end or 0))
        except Exception:
            pass
    return round(max(end_values), 3) if end_values else 0.0


def save_meeting_result(meeting: dict, segments: list[dict]) -> str:
    """保存或更新一次会议识别结果。返回 meeting_id。"""
    init_db()
    clean_segments = strip_audio_payload(segments)
    meeting_id = str(meeting.get("file_id") or meeting.get("id") or "").strip()
    if not meeting_id:
        raise ValueError("meeting file_id/id is required")

    now = _now()
    full_text = meeting.get("full_text") or build_full_text(clean_segments)
    duration_sec = meeting.get("duration_sec") or _duration_from_segments(clean_segments)

    with _connect() as conn:
        old = conn.execute("SELECT created_at FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        created_at = old["created_at"] if old else now
        conn.execute(
            """
            INSERT INTO meetings (
                id, filename, status, engine, created_at, updated_at, segments_count,
                duration_sec, full_text, summary_json, hotwords_json, speaker_stats_json,
                asr_optimizer_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename=excluded.filename,
                status=excluded.status,
                engine=excluded.engine,
                updated_at=excluded.updated_at,
                segments_count=excluded.segments_count,
                duration_sec=excluded.duration_sec,
                full_text=excluded.full_text,
                summary_json=excluded.summary_json,
                hotwords_json=excluded.hotwords_json,
                speaker_stats_json=excluded.speaker_stats_json,
                asr_optimizer_json=excluded.asr_optimizer_json,
                metadata_json=excluded.metadata_json
            """,
            (
                meeting_id,
                meeting.get("filename", ""),
                meeting.get("status", "completed"),
                meeting.get("engine", ""),
                created_at,
                now,
                int(meeting.get("segments_count") or len(clean_segments)),
                float(duration_sec or 0),
                full_text,
                _json_dumps(meeting.get("summary")),
                _json_dumps(meeting.get("hotwords") or []),
                _json_dumps(meeting.get("speaker_stats") or {}),
                _json_dumps(meeting.get("asr_optimizer")),
                _json_dumps({
                    "memory_id": meeting_id,
                    "source": meeting.get("source", "upload"),
                    "domain": meeting.get("domain"),
                }),
            ),
        )
        conn.execute("DELETE FROM segments WHERE meeting_id = ?", (meeting_id,))
        for idx, seg in enumerate(clean_segments, start=1):
            conn.execute(
                """
                INSERT INTO segments (
                    meeting_id, segment_index, speaker_id, start_sec, end_sec, text, raw_text,
                    asr_confidence, snr_db, quality_score, quality_label, terms_json,
                    data_points_json, corrections_json, logic_flags_json, uncertain_spans_json,
                    uncertainty_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meeting_id,
                    int(seg.get("index") or seg.get("segment_index") or idx),
                    seg.get("speaker_id") or seg.get("speaker") or "unknown",
                    float(seg.get("start_sec", seg.get("start", 0)) or 0),
                    float(seg.get("end_sec", seg.get("end", 0)) or 0),
                    seg.get("text") or seg.get("display_text") or "",
                    seg.get("raw_text") or "",
                    float(seg.get("asr_confidence") or 0),
                    float(seg.get("snr_db") or 0),
                    float(seg.get("quality_score") or 0),
                    seg.get("quality_label") or seg.get("quality") or "",
                    _json_dumps(seg.get("terms") or []),
                    _json_dumps(seg.get("data_points") or []),
                    _json_dumps(seg.get("corrections") or []),
                    _json_dumps(seg.get("logic_flags") or []),
                    _json_dumps(seg.get("uncertain_spans") or []),
                    _json_dumps(seg.get("uncertainty") or {}),
                    now,
                ),
            )
    return meeting_id


def _row_to_list_item(row: sqlite3.Row) -> dict:
    summary = _json_loads(row["summary_json"], {}) or {}
    summary_text = summary.get("summary") if isinstance(summary, dict) else ""
    if not summary_text:
        summary_text = (row["full_text"] or "").replace("\n", " ")[:120]
    full_text = row["full_text"] or ""
    return {
        "id": row["id"],
        "file_id": row["id"],
        "filename": row["filename"],
        "status": row["status"],
        "engine": row["engine"],
        "segments_count": row["segments_count"],
        "duration_sec": row["duration_sec"],
        "summary_preview": summary_text[:160] if summary_text else "",
        "full_text_preview": full_text[:1000],
        "hotwords": _json_loads(row["hotwords_json"], []) or [],
        "speaker_stats": _json_loads(row["speaker_stats_json"], {}) or {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_meetings(limit: int = 20, offset: int = 0, q: str = None) -> dict:
    """列出历史会议，可按文件名/全文/摘要/热词模糊搜索。"""
    init_db()
    limit = max(1, min(int(limit or 20), 100))
    offset = max(0, int(offset or 0))
    where = ""
    params = []
    if q:
        like = f"%{q}%"
        where = "WHERE filename LIKE ? OR full_text LIKE ? OR summary_json LIKE ? OR hotwords_json LIKE ?"
        params.extend([like, like, like, like])

    with _connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS c FROM meetings {where}", params).fetchone()["c"]
        rows = conn.execute(
            f"""
            SELECT * FROM meetings {where}
            ORDER BY COALESCE(updated_at, created_at) DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()
    return {
        "available": True,
        "items": [_row_to_list_item(r) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "q": q or "",
    }


def get_meeting(meeting_id: str) -> Optional[dict]:
    """获取单条历史会议详情。"""
    init_db()
    with _connect() as conn:
        meeting = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if not meeting:
            return None
        seg_rows = conn.execute(
            "SELECT * FROM segments WHERE meeting_id = ? ORDER BY segment_index ASC, id ASC",
            (meeting_id,),
        ).fetchall()

    segments = []
    for r in seg_rows:
        segments.append({
            "index": r["segment_index"],
            "speaker_id": r["speaker_id"],
            "start_sec": r["start_sec"],
            "end_sec": r["end_sec"],
            "text": r["text"],
            "raw_text": r["raw_text"],
            "asr_confidence": r["asr_confidence"],
            "snr_db": r["snr_db"],
            "quality_score": r["quality_score"],
            "quality_label": r["quality_label"],
            "terms": _json_loads(r["terms_json"], []) or [],
            "data_points": _json_loads(r["data_points_json"], []) or [],
            "corrections": _json_loads(r["corrections_json"], []) or [],
            "logic_flags": _json_loads(r["logic_flags_json"], []) or [],
            "uncertain_spans": _json_loads(r["uncertain_spans_json"], []) or [],
            "uncertainty": _json_loads(r["uncertainty_json"], {}) or {},
        })

    return {
        "id": meeting["id"],
        "file_id": meeting["id"],
        "memory_id": meeting["id"],
        "filename": meeting["filename"],
        "status": meeting["status"],
        "engine": meeting["engine"],
        "segments_count": meeting["segments_count"],
        "duration_sec": meeting["duration_sec"],
        "full_text": meeting["full_text"],
        "summary": _json_loads(meeting["summary_json"], None),
        "hotwords": _json_loads(meeting["hotwords_json"], []) or [],
        "speaker_stats": _json_loads(meeting["speaker_stats_json"], {}) or {},
        "asr_optimizer": _json_loads(meeting["asr_optimizer_json"], None),
        "metadata": _json_loads(meeting["metadata_json"], {}) or {},
        "segments": segments,
        "created_at": meeting["created_at"],
        "updated_at": meeting["updated_at"],
    }


def delete_meeting(meeting_id: str) -> bool:
    """删除一条历史会议。"""
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    return cur.rowcount > 0
