ALTER TABLE model_registry
    ADD COLUMN task VARCHAR(64) NOT NULL DEFAULT 'classification',
    ADD COLUMN tokenizer VARCHAR(128) NOT NULL DEFAULT 'indobenchmark/indobert-base-p1',
    ADD COLUMN calibration JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN dataset_split VARCHAR(64) NOT NULL DEFAULT 'train',
    ADD COLUMN eval_summary JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX ix_model_registry_task ON model_registry (task);

ALTER TABLE model_registry ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON model_registry FROM anon;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON model_registry FROM authenticated;
GRANT SELECT ON model_registry TO authenticated;

DROP POLICY IF EXISTS model_registry_service_role_only ON model_registry;
CREATE POLICY model_registry_service_role_only ON model_registry
    FOR ALL TO service_role USING (TRUE) WITH CHECK (TRUE);

DROP POLICY IF EXISTS model_registry_authenticated_select_only ON model_registry;
CREATE POLICY model_registry_authenticated_select_only ON model_registry
    FOR SELECT TO authenticated USING (TRUE);
