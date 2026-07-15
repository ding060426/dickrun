import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
DB_PATH = DATA_DIR / "diting.db"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
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
            """
        )
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
    allowed = ["display_name", "email", "phone", "role", "status"]
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
        query += " AND (organizer_user_id = ? OR participant_user_ids LIKE ?)"
        params.extend([user_id, f'%"{user_id}"%'])
    query += " ORDER BY start_time DESC"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [row_to_dict(row) for row in rows]


def create_reservation(data):
    title = (data.get("title") or "").strip()
    organizer_user_id = data.get("organizer_user_id")
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if not all([title, organizer_user_id, start_time, end_time]):
        raise ValueError("title, organizer_user_id, start_time and end_time are required")
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
    return get_reservation(reservation_id)


def get_reservation(reservation_id):
    with connect() as conn:
        return row_to_dict(conn.execute("SELECT * FROM meeting_reservations WHERE id = ?", (reservation_id,)).fetchone())


def update_reservation(reservation_id, data):
    allowed = ["title", "description", "participant_user_ids", "start_time", "end_time", "location", "meeting_type", "status"]
    fields = [field for field in allowed if field in data]
    if not fields:
        return get_reservation(reservation_id)
    values = []
    assignments = []
    for field in fields:
        assignments.append(f"{field} = ?")
        value = data[field]
        if field == "participant_user_ids" and isinstance(value, list):
            value = __import__("json").dumps(value)
        values.append(value)
    values.extend([now_iso(), reservation_id])
    with connect() as conn:
        conn.execute(f"UPDATE meeting_reservations SET {', '.join(assignments)}, updated_at = ? WHERE id = ?", values)
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
