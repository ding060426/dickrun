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
    avatar_data_url TEXT NOT NULL DEFAULT '',
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

-- 兼容已初始化的项目，可重复执行。
ALTER TABLE users
ADD COLUMN IF NOT EXISTS avatar_data_url TEXT NOT NULL DEFAULT '';

-- 预约参会成员（权威关系表；JSON 字段仅保留兼容性）
CREATE TABLE IF NOT EXISTS meeting_participants (
    meeting_id TEXT NOT NULL REFERENCES meeting_reservations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id),
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (meeting_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_reservations_organizer_time
ON meeting_reservations(organizer_user_id, start_time);

CREATE INDEX IF NOT EXISTS idx_meeting_participants_user
ON meeting_participants(user_id, meeting_id);

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

-- 从旧版 JSON 参会人字段迁移到关系表，可重复执行。
INSERT INTO meeting_participants (meeting_id, user_id, added_at)
SELECT reservation.id, participant.user_id, reservation.created_at
FROM meeting_reservations AS reservation
CROSS JOIN LATERAL jsonb_array_elements_text(reservation.participant_user_ids)
    AS participant(user_id)
JOIN users ON users.id = participant.user_id AND users.status = 'active'
WHERE participant.user_id <> reservation.organizer_user_id
ON CONFLICT (meeting_id, user_id) DO NOTHING;

-- 以后每次新增或修改预约时，在同一个 PostgreSQL 事务中校验并同步参会关系。
-- 这样不会出现预约保存成功但参会人日历尚未更新的中间状态。
CREATE OR REPLACE FUNCTION sync_meeting_participants_from_json()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF jsonb_typeof(NEW.participant_user_ids) <> 'array' THEN
        RAISE EXCEPTION 'participant_user_ids must be a JSON array';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(NEW.participant_user_ids) AS selected(user_id)
        LEFT JOIN users
          ON users.id = selected.user_id AND users.status = 'active'
        LEFT JOIN friends
          ON friends.user_id = NEW.organizer_user_id
         AND friends.friend_id = selected.user_id
        WHERE selected.user_id = NEW.organizer_user_id
           OR users.id IS NULL
           OR friends.friend_id IS NULL
    ) THEN
        RAISE EXCEPTION 'participants must be active colleagues of the organizer';
    END IF;

    DELETE FROM meeting_participants WHERE meeting_id = NEW.id;
    INSERT INTO meeting_participants (meeting_id, user_id, added_at)
    SELECT DISTINCT NEW.id, selected.user_id, now()
    FROM jsonb_array_elements_text(NEW.participant_user_ids) AS selected(user_id);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS meeting_reservations_sync_participants
ON meeting_reservations;
CREATE TRIGGER meeting_reservations_sync_participants
AFTER INSERT OR UPDATE OF participant_user_ids, organizer_user_id
ON meeting_reservations
FOR EACH ROW
EXECUTE FUNCTION sync_meeting_participants_from_json();

-- ============================================================
-- RLS 策略（开发阶段：所有表全开放）
-- ============================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS users_all ON users;
CREATE POLICY users_all ON users FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE auth_sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS auth_sessions_all ON auth_sessions;
CREATE POLICY auth_sessions_all ON auth_sessions FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_reservations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS meeting_reservations_all ON meeting_reservations;
CREATE POLICY meeting_reservations_all ON meeting_reservations FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_participants ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS meeting_participants_all ON meeting_participants;
CREATE POLICY meeting_participants_all ON meeting_participants FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_joins ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS meeting_joins_all ON meeting_joins;
CREATE POLICY meeting_joins_all ON meeting_joins FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE meeting_analyses ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS meeting_analyses_all ON meeting_analyses;
CREATE POLICY meeting_analyses_all ON meeting_analyses FOR ALL USING (true) WITH CHECK (true);

ALTER TABLE friends ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS friends_all ON friends;
CREATE POLICY friends_all ON friends FOR ALL USING (true) WITH CHECK (true);
