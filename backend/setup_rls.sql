-- 允许 anon key 对所有表进行读写（开发阶段）
-- 复制到 Supabase SQL Editor 执行

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
