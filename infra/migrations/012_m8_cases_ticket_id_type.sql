-- 012_m8_cases_ticket_id_type.sql
-- Relax ticket_id from UUID to VARCHAR(128) to allow custom and municipal ticket identifiers (e.g. TKT-...)

ALTER TABLE cases ALTER COLUMN ticket_id TYPE VARCHAR(128);
