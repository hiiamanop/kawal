CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS case_knowledge_bank (
    entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    case_id VARCHAR(128) NOT NULL REFERENCES cases(case_id),
    text_hash CHAR(64) NOT NULL,
    raw_text TEXT NOT NULL,
    embedding FLOAT8[] NOT NULL,
    category VARCHAR(64) NOT NULL,
    risk VARCHAR(32) NOT NULL,
    completeness VARCHAR(32) NOT NULL,
    location_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    root_cause_category VARCHAR(64),
    root_cause_summary TEXT,
    ticket_id VARCHAR(128),
    is_active_incident BOOLEAN NOT NULL DEFAULT TRUE,
    incident_cluster_id VARCHAR(128),
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_knowledge_bank_case UNIQUE (case_id)
);

CREATE INDEX IF NOT EXISTS ix_knowledge_bank_hash ON case_knowledge_bank (tenant_id, text_hash);
CREATE INDEX IF NOT EXISTS ix_knowledge_bank_active ON case_knowledge_bank (tenant_id, category, is_active_incident);
CREATE INDEX IF NOT EXISTS ix_knowledge_bank_cluster ON case_knowledge_bank (incident_cluster_id);
CREATE INDEX IF NOT EXISTS ix_knowledge_bank_created ON case_knowledge_bank (created_at DESC);

-- Portable dot-product cosine similarity for normalized embeddings
CREATE OR REPLACE FUNCTION cosine_similarity_float8(a float8[], b float8[]) RETURNS float8 AS $$
DECLARE
    dot float8 := 0;
    i int;
    len int;
BEGIN
    len := array_length(a, 1);
    IF len IS NULL OR len <> array_length(b, 1) THEN
        RETURN 0.0;
    END IF;
    FOR i IN 1..len LOOP
        dot := dot + a[i] * b[i];
    END LOOP;
    RETURN dot;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;
