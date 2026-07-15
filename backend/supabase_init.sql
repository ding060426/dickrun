-- ============================================================
-- 谛听 DiTing - Supabase 数据库初始化
-- 复制全部内容到 Supabase SQL Editor 执行
-- 项目: https://supabase.com/dashboard
-- ============================================================

-- 用户表
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

-- 认证会话表
CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ
);

-- 会议预约表
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

-- 参会记录表
CREATE TABLE IF NOT EXISTS meeting_joins (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meeting_reservations(id),
    user_id TEXT REFERENCES users(id),
    display_name TEXT NOT NULL,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 会议分析表
CREATE TABLE IF NOT EXISTS meeting_analyses (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
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

-- 好友关系表
CREATE TABLE IF NOT EXISTS friends (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    friend_id TEXT NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, friend_id)
);

-- ============================================================
-- RLS 策略（开发阶段：所有表全开放）
-- ============================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY users_all ON users FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE auth_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY auth_sessions_all ON auth_sessions FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_reservations ENABLE ROW LEVEL SECURITY;
CREATE POLICY meeting_reservations_all ON meeting_reservations FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_joins ENABLE ROW LEVEL SECURITY;
CREATE POLICY meeting_joins_all ON meeting_joins FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_analyses ENABLE ROW LEVEL SECURITY;
CREATE POLICY meeting_analyses_all ON meeting_analyses FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE friends ENABLE ROW LEVEL SECURITY;
CREATE POLICY friends_all ON friends FOR ALL USING (true) WITH CHECK (true);
