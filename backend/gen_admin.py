import hashlib, secrets, uuid
salt = secrets.token_hex(16)
h = hashlib.pbkdf2_hmac("sha256", "admin123".encode(), salt.encode(), 120000).hex()
uid = str(uuid.uuid4())
print(f"INSERT INTO users (id, username, display_name, role, status, password_salt, password_hash, email, phone, created_at, updated_at) VALUES ('{uid}', 'admin', 'Administrator', 'admin', 'active', '{salt}', '{h}', '', '', now(), now());")
