import hashlib
import hmac
import json
import os
import secrets
import uuid
from datetime import datetime, timezone

from supabase import create_client, Client


# Load .env file if present (before checking os.environ)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_supabase: Client = None


def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return salt, digest.hex()


def verify_password(password, salt, expected_hash):
    _, candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def init_db():
    """Ensure Supabase tables exist (run via Supabase SQL Editor)."""
    sql = """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        email TEXT DEFAULT '',
        phone TEXT DEFAULT '',
        role TEXT NOT NULL DEFAULT 'user',
        status TEXT NOT NULL DEFAULT 'active',
        password_salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS auth_sessions (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        expires_at TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS meeting_reservations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        organizer_user_id TEXT NOT NULL REFERENCES users(id),
        participant_user_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        start_time TIMESTAMPTZ NOT NULL,
        end_time TIMESTAMPTZ NOT NULL,
        location TEXT DEFAULT '',
        meeting_type TEXT NOT NULL DEFAULT 'offline',
        status TEXT NOT NULL DEFAULT 'scheduled',
        join_code TEXT NOT NULL UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS meeting_joins (
        id TEXT PRIMARY KEY,
        meeting_id TEXT NOT NULL REFERENCES meeting_reservations(id),
        user_id TEXT REFERENCES users(id),
        display_name TEXT NOT NULL,
        joined_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS meeting_analyses (
        id TEXT PRIMARY KEY,
        meeting_id TEXT NOT NULL REFERENCES meeting_reservations(id),
        title TEXT NOT NULL,
        transcript_json JSONB NOT NULL DEFAULT '[]'::jsonb,
        segments_count INTEGER NOT NULL DEFAULT 0,
        duration_sec REAL NOT NULL DEFAULT 0,
        logic_flags_count INTEGER NOT NULL DEFAULT 0,
        low_confidence_count INTEGER NOT NULL DEFAULT 0,
        corrections_count INTEGER NOT NULL DEFAULT 0,
        overall_confidence REAL DEFAULT 0,
        hotwords JSONB NOT NULL DEFAULT '[]'::jsonb,
        summary_json JSONB DEFAULT '{}'::jsonb,
        created_by TEXT REFERENCES users(id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS friends (
        id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
        user_id TEXT NOT NULL REFERENCES users(id),
        friend_id TEXT NOT NULL REFERENCES users(id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (user_id, friend_id),
        CHECK (user_id <> friend_id)
    );
    """
    print("[Supabase] Please run this SQL in your Supabase SQL Editor to create tables.")
    print("[Supabase] Alternatively, tables will be auto-created on first insert.")
    return sql


# ── Users ─────────────────────────────────────────────────────

def _clean_user(row):
    if not row:
        return None
    data = dict(row)
    data.pop("password_salt", None)
    data.pop("password_hash", None)
    return data


def list_users():
    client = _get_client()
    result = client.table("users").select("*").neq("status", "deleted").order("created_at", desc=True).execute()
    return [_clean_user(r) for r in (result.data or [])]


def get_user(user_id):
    client = _get_client()
    result = client.table("users").select("*").eq("id", user_id).execute()
    return _clean_user(result.data[0]) if result.data else None


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
    client = _get_client()
    client.table("users").insert({
        "id": user_id, "username": username, "display_name": display_name,
        "email": data.get("email") or "", "phone": data.get("phone") or "",
        "role": data.get("role") or "user", "status": data.get("status") or "active",
        "password_salt": salt, "password_hash": password_hash,
        "created_at": timestamp, "updated_at": timestamp,
    }).execute()
    return get_user(user_id)


def update_user(user_id, data):
    allowed = ["display_name", "email", "phone", "role", "status"]
    fields = {field: data[field] for field in allowed if field in data}
    if not fields:
        return get_user(user_id)
    fields["updated_at"] = now_iso()
    client = _get_client()
    client.table("users").update(fields).eq("id", user_id).execute()
    return get_user(user_id)


# ── Auth ──────────────────────────────────────────────────────

def login(username, password):
    client = _get_client()
    result = client.table("users").select("*").eq("username", username).eq("status", "active").execute()
    if not result.data:
        return None
    user = result.data[0]
    if not verify_password(password, user["password_salt"], user["password_hash"]):
        return None
    token = secrets.token_urlsafe(32)
    client.table("auth_sessions").insert({
        "token": token, "user_id": user["id"],
        "created_at": now_iso(), "expires_at": None,
    }).execute()
    return {"token": token, "user": _clean_user(user)}


def get_user_by_token(token):
    if not token:
        return None
    client = _get_client()
    result = client.table("auth_sessions").select("*, users(*)").eq("token", token).execute()
    if not result.data:
        return None
    session = result.data[0]
    user = session.get("users")
    if not user or user.get("status") != "active":
        return None
    return _clean_user(user)


def logout(token):
    if not token:
        return True
    client = _get_client()
    client.table("auth_sessions").delete().eq("token", token).execute()
    return True


# ── Reservations ──────────────────────────────────────────────

def list_reservations(user_id=None, status=None):
    client = _get_client()
    query = client.table("meeting_reservations").select("*").order("start_time", desc=True)
    if status:
        query = query.eq("status", status)
    if user_id:
        query = query.or_(f"organizer_user_id.eq.{user_id},participant_user_ids.cs.{{\"{user_id}\"}}")
    result = query.execute()
    return result.data or []


def get_reservation(reservation_id):
    client = _get_client()
    result = client.table("meeting_reservations").select("*").eq("id", reservation_id).execute()
    return result.data[0] if result.data else None


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

    client = _get_client()
    client.table("meeting_reservations").insert({
        "id": reservation_id, "title": title,
        "description": data.get("description") or "",
        "organizer_user_id": organizer_user_id,
        "participant_user_ids": participant_ids,
        "start_time": start_time, "end_time": end_time,
        "location": data.get("location") or "",
        "meeting_type": data.get("meeting_type") or "offline",
        "status": data.get("status") or "scheduled",
        "join_code": join_code,
        "created_at": timestamp, "updated_at": timestamp,
    }).execute()
    return get_reservation(reservation_id)


def update_reservation(reservation_id, data):
    allowed = ["title", "description", "participant_user_ids", "start_time", "end_time", "location", "meeting_type", "status"]
    fields = {field: data[field] for field in allowed if field in data}
    if not fields:
        return get_reservation(reservation_id)
    fields["updated_at"] = now_iso()
    client = _get_client()
    client.table("meeting_reservations").update(fields).eq("id", reservation_id).execute()
    return get_reservation(reservation_id)


def join_meeting(data):
    meeting_id = data.get("meeting_id")
    join_code = data.get("join_code")
    user_id = data.get("user_id")
    display_name = (data.get("display_name") or "Guest").strip()

    client = _get_client()
    if meeting_id:
        result = client.table("meeting_reservations").select("*").eq("id", meeting_id).execute()
    else:
        result = client.table("meeting_reservations").select("*").eq("join_code", join_code).execute()
    if not result.data:
        raise ValueError("meeting not found")
    meeting = result.data[0]
    join_id = str(uuid.uuid4())
    client.table("meeting_joins").insert({
        "id": join_id, "meeting_id": meeting["id"],
        "user_id": user_id, "display_name": display_name,
        "joined_at": now_iso(),
    }).execute()
    return {"join_id": join_id, "meeting": meeting}


# ── Meeting Analysis ──────────────────────────────────────────

def save_analysis(data):
    analysis_id = str(uuid.uuid4())
    timestamp = now_iso()
    client = _get_client()
    client.table("meeting_analyses").insert({
        "id": analysis_id,
        "meeting_id": data.get("meeting_id") or analysis_id,
        "title": data.get("title") or "Untitled",
        "transcript_json": data.get("transcript_json") or [],
        "segments_count": data.get("segments_count") or 0,
        "duration_sec": data.get("duration_sec") or 0,
        "logic_flags_count": data.get("logic_flags_count") or 0,
        "low_confidence_count": data.get("low_confidence_count") or 0,
        "corrections_count": data.get("corrections_count") or 0,
        "overall_confidence": data.get("overall_confidence") or 0,
        "hotwords": data.get("hotwords") or [],
        "summary_json": data.get("summary_json") or {},
        "created_by": data.get("created_by"),
        "created_at": timestamp,
    }).execute()
    return get_analysis(analysis_id)


def get_analysis(analysis_id):
    client = _get_client()
    result = client.table("meeting_analyses").select("*").eq("id", analysis_id).execute()
    return result.data[0] if result.data else None


def list_analyses(user_id=None, meeting_id=None, limit=50):
    client = _get_client()
    query = client.table("meeting_analyses").select("*").order("created_at", desc=True).limit(limit)
    if user_id:
        query = query.eq("created_by", user_id)
    if meeting_id:
        query = query.eq("meeting_id", meeting_id)
    result = query.execute()
    return result.data or []


def delete_analysis(analysis_id):
    client = _get_client()
    client.table("meeting_analyses").delete().eq("id", analysis_id).execute()
    return True


# ── Friends ───────────────────────────────────────────────────

def search_users(keyword, exclude_user_id=None, limit=20):
    client = _get_client()
    query = client.table("users").select("id,username,display_name,role,status").ilike("username", f"%{keyword}%").neq("status", "deleted").limit(limit)
    if exclude_user_id:
        query = query.neq("id", exclude_user_id)
    result = query.execute()
    return result.data or []


def list_friends(user_id):
    client = _get_client()
    result = client.table("friends").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    friends = []
    seen = set()
    for r in (result.data or []):
        fid = r["friend_id"]
        if fid in seen:
            continue
        seen.add(fid)
        u = client.table("users").select("username,display_name,role").eq("id", fid).execute()
        info = u.data[0] if u.data else {}
        friends.append({
            "id": r["id"],
            "friend_id": fid,
            "username": info.get("username", ""),
            "display_name": info.get("display_name", ""),
            "role": info.get("role", ""),
            "created_at": r["created_at"],
        })
    return friends


def add_friend(user_id, friend_id):
    client = _get_client()
    if user_id == friend_id:
        return None
    try:
        client.table("friends").insert({"user_id": user_id, "friend_id": friend_id}).execute()
        return {"user_id": user_id, "friend_id": friend_id}
    except Exception as exc:
        print(f"[Supabase] add_friend failed: {exc}")
        return None


def remove_friend(user_id, friend_id):
    client = _get_client()
    client.table("friends").delete().eq("user_id", user_id).eq("friend_id", friend_id).execute()
    return True
