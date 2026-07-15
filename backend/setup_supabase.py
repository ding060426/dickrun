import os, sys
os.chdir(r"C:\Users\98068\Desktop\dickrun-new-meeting\backend")
sys.path.insert(0, ".")

# Load .env
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

from supabase import create_client
import hashlib, secrets, uuid
from datetime import datetime, timezone

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
c = create_client(url, key)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return salt, digest.hex()

# Create admin user
existing = c.table("users").select("id").eq("username", "admin").execute()
if existing.data:
    print("admin user already exists, skipping")
else:
    salt, password_hash = hash_password("admin123")
    uid = str(uuid.uuid4())
    ts = now_iso()
    c.table("users").insert({
        "id": uid,
        "username": "admin",
        "display_name": "Administrator",
        "role": "admin",
        "status": "active",
        "password_salt": salt,
        "password_hash": password_hash,
        "email": "",
        "phone": "",
        "created_at": ts,
        "updated_at": ts,
    }).execute()
    print(f"admin user created: admin / admin123")

# Verify all tables
for t in ["users", "auth_sessions", "meeting_reservations", "meeting_joins", "meeting_analyses", "friends"]:
    try:
        c.table(t).select("*").limit(1).execute()
        print(f"  {t}: OK")
    except Exception as e:
        print(f"  {t}: MISSING - {e}")

print("\nSetup complete! Run: python start.py")
