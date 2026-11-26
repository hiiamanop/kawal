CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS harness_case_cache (
    cache_key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    analysis_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    text_hash CHAR(64) NOT NULL,
    source_post_id TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    category TEXT NOT NULL,
    rule_category TEXT NOT NULL,
    source_intent TEXT NOT NULL,
    location_quality TEXT NOT NULL,
    location_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_spans JSONB NOT NULL DEFAULT '[]'::jsonb,
    analysis_payload JSONB NOT NULL,
    record_guards JSONB NOT NULL DEFAULT '{}'::jsonb,
    trust_decision TEXT NOT NULL,
    status TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    last_hit_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_harness_cache_text
    ON harness_case_cache (namespace, analysis_version, embedding_model, text_hash);

CREATE INDEX IF NOT EXISTS ix_harness_cache_lookup
    ON harness_case_cache (namespace, analysis_version, embedding_model, category, source_intent);

CREATE INDEX IF NOT EXISTS ix_harness_cache_hnsw
    ON harness_case_cache USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
