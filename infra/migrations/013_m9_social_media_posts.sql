CREATE TABLE IF NOT EXISTS social_media_posts (
    post_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(32) NOT NULL,
    external_id VARCHAR(256) NOT NULL,
    query TEXT,
    url TEXT,
    author VARCHAR(256),
    text TEXT NOT NULL CHECK (length(text) > 0),
    published_at TIMESTAMPTZ,
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    case_id VARCHAR(128) REFERENCES cases(case_id) ON DELETE SET NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_social_media_post UNIQUE (platform, external_id)
);

CREATE INDEX IF NOT EXISTS ix_social_media_posts_fetched ON social_media_posts (fetched_at DESC);
CREATE INDEX IF NOT EXISTS ix_social_media_posts_platform ON social_media_posts (platform);
CREATE INDEX IF NOT EXISTS ix_social_media_posts_case ON social_media_posts (case_id);
