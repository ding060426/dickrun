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


def _participant_ids_from_value(value):
    if not isinstance(value, list):
        try:
            value = json.loads(value or "[]")
        except (TypeError, json.JSONDecodeError):
            value = []
    return list(dict.fromkeys(str(item) for item in value if item))


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


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return salt, digest.hex()


def verify_password(password, salt, expected_hash):
    _, candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def init_db():
    """Return the authoritative schema that must be run in Supabase SQL Editor."""
    schema_path = os.path.join(os.path.dirname(__file__), "..", "supabase_init.sql")
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        sql = schema_file.read()
    print(f"[Supabase] Run the schema in SQL Editor before startup: {schema_path}")
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

def _validate_colleague_participants(client, organizer_user_id, participant_ids):
    normalized = [
        user_id
        for user_id in _participant_ids_from_value(participant_ids)
        if user_id != organizer_user_id
    ]
    if not normalized:
        return []
    active = client.table("users").select("id").eq("status", "active").in_("id", normalized).execute()
    colleagues = client.table("friends").select("friend_id").eq("user_id", organizer_user_id).in_("friend_id", normalized).execute()
    active_ids = {row["id"] for row in (active.data or [])}
    colleague_ids = {row["friend_id"] for row in (colleagues.data or [])}
    if set(normalized) - active_ids.intersection(colleague_ids):
        raise ValueError("participants must be active colleagues of the organizer")
    return normalized


def _with_reservation_people(client, reservation):
    if not reservation:
        return None
    data = dict(reservation)
    organizer = client.table("users").select("id,username,display_name,role,status").eq("id", data["organizer_user_id"]).execute()
    membership = client.table("meeting_participants").select("user_id").eq("meeting_id", data["id"]).execute()
    participant_ids = [row["user_id"] for row in (membership.data or [])]
    participants = []
    if participant_ids:
        result = client.table("users").select("id,username,display_name,role,status").eq("status", "active").in_("id", participant_ids).execute()
        by_id = {row["id"]: row for row in (result.data or [])}
        participants = [by_id[user_id] for user_id in participant_ids if user_id in by_id]
        participants.sort(key=lambda item: (item.get("display_name") or item.get("username") or ""))
    data["organizer"] = organizer.data[0] if organizer.data else None
    data["participants"] = participants
    data["participant_user_ids"] = [item["id"] for item in participants]
    return data

def list_reservations(user_id=None, status=None):
    client = _get_client()
    query = client.table("meeting_reservations").select("*").order("start_time", desc=True)
    if status:
        query = query.eq("status", status)
    if user_id:
        memberships = client.table("meeting_participants").select("meeting_id").eq("user_id", user_id).execute()
        meeting_ids = [row["meeting_id"] for row in (memberships.data or [])]
        if meeting_ids:
            query = query.or_(
                f"organizer_user_id.eq.{user_id},id.in.({','.join(meeting_ids)})"
            )
        else:
            query = query.eq("organizer_user_id", user_id)
    result = query.execute()
    return [_with_reservation_people(client, row) for row in (result.data or [])]


def get_reservation(reservation_id):
    client = _get_client()
    result = client.table("meeting_reservations").select("*").eq("id", reservation_id).execute()
    return _with_reservation_people(client, result.data[0]) if result.data else None


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

    client = _get_client()
    organizer = get_user(organizer_user_id)
    if not organizer or organizer.get("status") != "active":
        raise ValueError("organizer_user_id is invalid")
    participant_ids = _validate_colleague_participants(
        client, organizer_user_id, participant_ids
    )
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
    # The PostgreSQL trigger in supabase_init.sql synchronizes the normalized
    # membership table in the same transaction as this reservation insert.
    return get_reservation(reservation_id)


def update_reservation(reservation_id, data):
    allowed = ["title", "description", "participant_user_ids", "start_time", "end_time", "location", "meeting_type", "status"]
    fields = {field: data[field] for field in allowed if field in data}
    if not fields:
        return get_reservation(reservation_id)
    client = _get_client()
    existing = get_reservation(reservation_id)
    if not existing:
        return None
    _validate_reservation_times(
        fields.get("start_time", existing["start_time"]),
        fields.get("end_time", existing["end_time"]),
    )
    participant_ids = None
    if "participant_user_ids" in fields:
        if not isinstance(fields["participant_user_ids"], list):
            raise ValueError("participant_user_ids must be a list")
        participant_ids = _validate_colleague_participants(
            client,
            existing["organizer_user_id"],
            fields["participant_user_ids"],
        )
        fields["participant_user_ids"] = participant_ids
    fields["updated_at"] = now_iso()
    client.table("meeting_reservations").update(fields).eq("id", reservation_id).execute()
    # Membership synchronization is atomic through the PostgreSQL trigger.
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
