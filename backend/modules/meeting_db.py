import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(
    os.environ.get("DITING_MANAGEMENT_DB_PATH", BACKEND_DIR / "data" / "diting.db")
)
DATA_DIR = DB_PATH.parent


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row):
    return dict(row) if row else None


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return salt, digest.hex()


def verify_password(password, salt, expected_hash):
    _, candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def init_db():
    with connect() as conn:
        # WAL lets calendar reads continue while another local account writes.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                avatar_data_url TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'active',
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS meeting_reservations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                organizer_user_id TEXT NOT NULL,
                participant_user_ids TEXT NOT NULL DEFAULT '[]',
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                location TEXT,
                meeting_type TEXT NOT NULL DEFAULT 'offline',
                status TEXT NOT NULL DEFAULT 'scheduled',
                join_code TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (organizer_user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS meeting_joins (
                id TEXT PRIMARY KEY,
                meeting_id TEXT NOT NULL,
                user_id TEXT,
                display_name TEXT NOT NULL,
                joined_at TEXT NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meeting_reservations(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS meeting_participants (
                meeting_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (meeting_id, user_id),
                FOREIGN KEY (meeting_id) REFERENCES meeting_reservations(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS meeting_analyses (
                id TEXT PRIMARY KEY,
                meeting_id TEXT NOT NULL,
                title TEXT NOT NULL,
                transcript_json TEXT NOT NULL DEFAULT '[]',
                segments_count INTEGER NOT NULL DEFAULT 0,
                duration_sec REAL NOT NULL DEFAULT 0,
                logic_flags_count INTEGER NOT NULL DEFAULT 0,
                low_confidence_count INTEGER NOT NULL DEFAULT 0,
                corrections_count INTEGER NOT NULL DEFAULT 0,
                overall_confidence REAL DEFAULT 0,
                hotwords TEXT NOT NULL DEFAULT '[]',
                summary_json TEXT NOT NULL DEFAULT '{}',
                created_by TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (created_by) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS friends (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                friend_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (user_id, friend_id),
                CHECK (user_id <> friend_id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (friend_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_reservations_organizer_time
            ON meeting_reservations(organizer_user_id, start_time);
            CREATE INDEX IF NOT EXISTS idx_meeting_participants_user
            ON meeting_participants(user_id, meeting_id);
            """
        )
        user_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "avatar_data_url" not in user_columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN avatar_data_url TEXT NOT NULL DEFAULT ''"
            )
        _migrate_reservation_participants(conn)
        existing = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if not existing:
            salt, password_hash = hash_password(os.environ.get("DITING_ADMIN_PASSWORD", "admin123"))
            timestamp = now_iso()
            conn.execute(
                """
                INSERT INTO users (
                    id, username, display_name, email, phone, role, status,
                    password_salt, password_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), "admin", "Administrator", "", "", "admin", "active",
                    salt, password_hash, timestamp, timestamp,
                ),
            )


def public_user(row):
    data = row_to_dict(row)
    if not data:
        return None
    data.pop("password_salt", None)
    data.pop("password_hash", None)
    return data


def _participant_ids_from_value(value):
    if isinstance(value, list):
        values = value
    else:
        try:
            values = json.loads(value or "[]")
        except (TypeError, json.JSONDecodeError):
            values = []
    return list(dict.fromkeys(str(item) for item in values if item))


def _validate_reservation_times(start_time, end_time):
    try:
        start = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ValueError("start_time and end_time must be ISO datetimes") from error
    local_timezone = datetime.now().astimezone().tzinfo
    if start.tzinfo is None:
        start = start.replace(tzinfo=local_timezone)
    if end.tzinfo is None:
        end = end.replace(tzinfo=local_timezone)
    if end <= start:
        raise ValueError("end_time must be after start_time")


def _migrate_reservation_participants(conn):
    rows = conn.execute(
        "SELECT id, organizer_user_id, participant_user_ids, created_at FROM meeting_reservations"
    ).fetchall()
    for row in rows:
        for user_id in _participant_ids_from_value(row["participant_user_ids"]):
            if user_id == row["organizer_user_id"]:
                continue
            user = conn.execute(
                "SELECT id FROM users WHERE id = ? AND status = 'active'",
                (user_id,),
            ).fetchone()
            if user:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO meeting_participants (meeting_id, user_id, added_at)
                    VALUES (?, ?, ?)
                    """,
                    (row["id"], user_id, row["created_at"]),
                )


def _validate_colleague_participants(conn, organizer_user_id, participant_ids):
    normalized = [
        user_id
        for user_id in _participant_ids_from_value(participant_ids)
        if user_id != organizer_user_id
    ]
    if not normalized:
        return []
    placeholders = ",".join("?" for _ in normalized)
    active_rows = conn.execute(
        f"SELECT id FROM users WHERE status = 'active' AND id IN ({placeholders})",
        normalized,
    ).fetchall()
    active_ids = {row["id"] for row in active_rows}
    colleague_rows = conn.execute(
        f"""
        SELECT friend_id FROM friends
        WHERE user_id = ? AND friend_id IN ({placeholders})
        """,
        [organizer_user_id, *normalized],
    ).fetchall()
    colleague_ids = {row["friend_id"] for row in colleague_rows}
    invalid_ids = set(normalized) - active_ids.intersection(colleague_ids)
    if invalid_ids:
        raise ValueError("participants must be active colleagues of the organizer")
    return normalized


def _reservation_to_dict(conn, row):
    reservation = row_to_dict(row)
    if not reservation:
        return None
    organizer = conn.execute(
        """
        SELECT id, username, display_name, role, status
        FROM users WHERE id = ?
        """,
        (reservation["organizer_user_id"],),
    ).fetchone()
    participants = conn.execute(
        """
        SELECT users.id, users.username, users.display_name, users.role, users.status
        FROM meeting_participants
        JOIN users ON users.id = meeting_participants.user_id
        WHERE meeting_participants.meeting_id = ? AND users.status = 'active'
        ORDER BY users.display_name, users.username
        """,
        (reservation["id"],),
    ).fetchall()
    reservation["organizer"] = row_to_dict(organizer)
    reservation["participants"] = [row_to_dict(item) for item in participants]
    reservation["participant_user_ids"] = [item["id"] for item in reservation["participants"]]
    return reservation


def list_users():
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE status != 'deleted' ORDER BY created_at DESC"
        ).fetchall()
        return [public_user(row) for row in rows]


def get_user(user_id):
    with connect() as conn:
        return public_user(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def create_user(data):
    username = (data.get("username") or "").strip()
    display_name = (data.get("display_name") or username).strip()
    password = data.get("password") or "123456"
    if not username:
        raise ValueError("username is required")
    if len(password) < 6:
        raise ValueError("password must be at least 6 characters")

    salt, password_hash = hash_password(password)
    user_id = str(uuid.uuid4())
    timestamp = now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO users (
                id, username, display_name, email, phone, role, status,
                password_salt, password_hash, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, username, display_name, data.get("email") or "", data.get("phone") or "",
                data.get("role") or "user", data.get("status") or "active",
                salt, password_hash, timestamp, timestamp,
            ),
        )
    return get_user(user_id)


def update_user(user_id, data):
    allowed = ["display_name", "email", "phone", "avatar_data_url", "role", "status"]
    fields = [field for field in allowed if field in data]
    if not fields:
        return get_user(user_id)
    assignments = ", ".join(f"{field} = ?" for field in fields)
    values = [data[field] for field in fields]
    values.extend([now_iso(), user_id])
    with connect() as conn:
        conn.execute(f"UPDATE users SET {assignments}, updated_at = ? WHERE id = ?", values)
    return get_user(user_id)


def login(username, password):
    with connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ? AND status = 'active'", (username,)).fetchone()
        if not user or not verify_password(password, user["password_salt"], user["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO auth_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user["id"], now_iso(), None),
        )
        return {"token": token, "user": public_user(user)}


def get_user_by_token(token):
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            """
            SELECT users.* FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token = ? AND users.status = 'active'
            """,
            (token,),
        ).fetchone()
        return public_user(row)


def logout(token):
    with connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))


# ── Friends ───────────────────────────────────────────────────

def search_users(keyword, exclude_user_id=None, limit=20):
    keyword = f"%{(keyword or '').strip()}%"
    query = """
        SELECT id, username, display_name, role, status
        FROM users
        WHERE status != 'deleted' AND username LIKE ?
    """
    params = [keyword]
    if exclude_user_id:
        query += " AND id != ?"
        params.append(exclude_user_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [row_to_dict(row) for row in rows]


def list_friends(user_id):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT friends.id, friends.friend_id, friends.created_at,
                   users.username, users.display_name, users.role
            FROM friends
            JOIN users ON users.id = friends.friend_id
            WHERE friends.user_id = ? AND users.status != 'deleted'
            ORDER BY friends.created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def add_friend(user_id, friend_id):
    if user_id == friend_id:
        return None
    friend_row_id = str(uuid.uuid4())
    timestamp = now_iso()
    try:
        with connect() as conn:
            user = conn.execute("SELECT id FROM users WHERE id = ? AND status = 'active'", (friend_id,)).fetchone()
            if not user:
                return None
            conn.execute(
                "INSERT INTO friends (id, user_id, friend_id, created_at) VALUES (?, ?, ?, ?)",
                (friend_row_id, user_id, friend_id, timestamp),
            )
            row = conn.execute("SELECT * FROM friends WHERE id = ?", (friend_row_id,)).fetchone()
            return row_to_dict(row)
    except sqlite3.IntegrityError:
        return None


def remove_friend(user_id, friend_id):
    with connect() as conn:
        conn.execute("DELETE FROM friends WHERE user_id = ? AND friend_id = ?", (user_id, friend_id))
    return True


def list_reservations(user_id=None, status=None):
    query = "SELECT * FROM meeting_reservations WHERE 1 = 1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if user_id:
        query += """
            AND (
                organizer_user_id = ?
                OR EXISTS (
                    SELECT 1 FROM meeting_participants
                    WHERE meeting_participants.meeting_id = meeting_reservations.id
                      AND meeting_participants.user_id = ?
                )
            )
        """
        params.extend([user_id, user_id])
    query += " ORDER BY start_time DESC"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [_reservation_to_dict(conn, row) for row in rows]


def create_reservation(data):
    title = (data.get("title") or "").strip()
    organizer_user_id = data.get("organizer_user_id")
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if not all([title, organizer_user_id, start_time, end_time]):
        raise ValueError("title, organizer_user_id, start_time and end_time are required")
    _validate_reservation_times(start_time, end_time)
    participant_ids = data.get("participant_user_ids") or []
    if not isinstance(participant_ids, list):
        raise ValueError("participant_user_ids must be a list")
    reservation_id = str(uuid.uuid4())
    timestamp = now_iso()
    join_code = secrets.token_urlsafe(8)
    with connect() as conn:
        user = conn.execute("SELECT id FROM users WHERE id = ? AND status = 'active'", (organizer_user_id,)).fetchone()
        if not user:
            raise ValueError("organizer_user_id is invalid")
        participant_ids = _validate_colleague_participants(
            conn, organizer_user_id, participant_ids
        )
        conn.execute(
            """
            INSERT INTO meeting_reservations (
                id, title, description, organizer_user_id, participant_user_ids,
                start_time, end_time, location, meeting_type, status, join_code, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reservation_id, title, data.get("description") or "", organizer_user_id,
                __import__("json").dumps(participant_ids), start_time, end_time,
                data.get("location") or "", data.get("meeting_type") or "offline",
                data.get("status") or "scheduled", join_code, timestamp, timestamp,
            ),
        )
        conn.executemany(
            """
            INSERT INTO meeting_participants (meeting_id, user_id, added_at)
            VALUES (?, ?, ?)
            """,
            [(reservation_id, user_id, timestamp) for user_id in participant_ids],
        )
    return get_reservation(reservation_id)


def get_reservation(reservation_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM meeting_reservations WHERE id = ?", (reservation_id,)
        ).fetchone()
        return _reservation_to_dict(conn, row)


def update_reservation(reservation_id, data):
    allowed = ["title", "description", "participant_user_ids", "start_time", "end_time", "location", "meeting_type", "status"]
    fields = [field for field in allowed if field in data]
    if not fields:
        return get_reservation(reservation_id)
    with connect() as conn:
        existing = conn.execute(
            "SELECT * FROM meeting_reservations WHERE id = ?", (reservation_id,)
        ).fetchone()
        if not existing:
            return None
        _validate_reservation_times(
            data.get("start_time", existing["start_time"]),
            data.get("end_time", existing["end_time"]),
        )
        values = []
        assignments = []
        participant_ids = None
        for field in fields:
            assignments.append(f"{field} = ?")
            value = data[field]
            if field == "participant_user_ids":
                if not isinstance(value, list):
                    raise ValueError("participant_user_ids must be a list")
                participant_ids = _validate_colleague_participants(
                    conn, existing["organizer_user_id"], value
                )
                value = json.dumps(participant_ids)
            values.append(value)
        values.extend([now_iso(), reservation_id])
        conn.execute(f"UPDATE meeting_reservations SET {', '.join(assignments)}, updated_at = ? WHERE id = ?", values)
        if participant_ids is not None:
            conn.execute(
                "DELETE FROM meeting_participants WHERE meeting_id = ?",
                (reservation_id,),
            )
            conn.executemany(
                """
                INSERT INTO meeting_participants (meeting_id, user_id, added_at)
                VALUES (?, ?, ?)
                """,
                [
                    (reservation_id, user_id, now_iso())
                    for user_id in participant_ids
                ],
            )
    return get_reservation(reservation_id)


def join_meeting(data):
    meeting_id = data.get("meeting_id")
    join_code = data.get("join_code")
    user_id = data.get("user_id")
    display_name = (data.get("display_name") or "Guest").strip()
    with connect() as conn:
        if meeting_id:
            meeting = conn.execute("SELECT * FROM meeting_reservations WHERE id = ?", (meeting_id,)).fetchone()
        else:
            meeting = conn.execute("SELECT * FROM meeting_reservations WHERE join_code = ?", (join_code,)).fetchone()
        if not meeting:
            raise ValueError("meeting not found")
        join_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO meeting_joins (id, meeting_id, user_id, display_name, joined_at) VALUES (?, ?, ?, ?, ?)",
            (join_id, meeting["id"], user_id, display_name, now_iso()),
        )
        return {"join_id": join_id, "meeting": row_to_dict(meeting)}


def _analysis_to_dict(row):
    analysis = row_to_dict(row)
    if not analysis:
        return None
    for field, fallback in (
        ("transcript_json", []),
        ("hotwords", []),
        ("summary_json", {}),
    ):
        try:
            analysis[field] = json.loads(analysis.get(field) or "")
        except (TypeError, json.JSONDecodeError):
            analysis[field] = fallback
    return analysis


def save_analysis(data):
    analysis_id = str(uuid.uuid4())
    values = {
        "id": analysis_id,
        "meeting_id": data.get("meeting_id") or analysis_id,
        "title": data.get("title") or "Untitled",
        "transcript_json": json.dumps(data.get("transcript_json") or [], ensure_ascii=False),
        "segments_count": data.get("segments_count") or 0,
        "duration_sec": data.get("duration_sec") or 0,
        "logic_flags_count": data.get("logic_flags_count") or 0,
        "low_confidence_count": data.get("low_confidence_count") or 0,
        "corrections_count": data.get("corrections_count") or 0,
        "overall_confidence": data.get("overall_confidence") or 0,
        "hotwords": json.dumps(data.get("hotwords") or [], ensure_ascii=False),
        "summary_json": json.dumps(data.get("summary_json") or {}, ensure_ascii=False),
        "created_by": data.get("created_by"),
        "created_at": now_iso(),
    }
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO meeting_analyses (
                id, meeting_id, title, transcript_json, segments_count,
                duration_sec, logic_flags_count, low_confidence_count,
                corrections_count, overall_confidence, hotwords,
                summary_json, created_by, created_at
            ) VALUES (
                :id, :meeting_id, :title, :transcript_json, :segments_count,
                :duration_sec, :logic_flags_count, :low_confidence_count,
                :corrections_count, :overall_confidence, :hotwords,
                :summary_json, :created_by, :created_at
            )
            """,
            values,
        )
    return get_analysis(analysis_id)


def get_analysis(analysis_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM meeting_analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    return _analysis_to_dict(row)


def list_analyses(user_id=None, meeting_id=None, limit=50):
    query = "SELECT * FROM meeting_analyses WHERE 1 = 1"
    params = []
    if user_id:
        query += " AND created_by = ?"
        params.append(user_id)
    if meeting_id:
        query += " AND meeting_id = ?"
        params.append(meeting_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_analysis_to_dict(row) for row in rows]


def delete_analysis(analysis_id):
    with connect() as conn:
        conn.execute("DELETE FROM meeting_analyses WHERE id = ?", (analysis_id,))
    return True
