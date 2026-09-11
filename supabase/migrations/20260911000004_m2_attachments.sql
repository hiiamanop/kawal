CREATE OR REPLACE FUNCTION public.kawal_attachment_path_is_valid(path TEXT)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
AS $$ SELECT path <> '' AND path !~ '(^|/)\.\.?(/|$)' $$;

ALTER TABLE attachments
    ADD COLUMN storage_bucket VARCHAR(64) NOT NULL DEFAULT 'attachments',
    ADD COLUMN is_private BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN retention_class VARCHAR(64) NOT NULL DEFAULT 'evidence';

ALTER TABLE attachments
    ADD CONSTRAINT ck_attachments_mime_type
    CHECK (mime_type IN ('image/jpeg', 'image/png', 'image/webp')),
    ADD CONSTRAINT ck_attachments_private_storage
    CHECK (storage_bucket = 'attachments' AND is_private),
    ADD CONSTRAINT ck_attachments_content_hash
    CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ck_attachments_object_key
    CHECK (public.kawal_attachment_path_is_valid(object_key));

DROP POLICY IF EXISTS attachments_service_role_only ON attachments;
CREATE POLICY attachments_service_role_only ON attachments
    FOR ALL TO service_role USING (TRUE) WITH CHECK (TRUE);

DROP POLICY IF EXISTS storage_attachments_service_role_only ON storage.objects;
CREATE POLICY storage_attachments_service_role_only ON storage.objects
    FOR ALL TO service_role USING (bucket_id = 'attachments')
    WITH CHECK (bucket_id = 'attachments');

CREATE INDEX ix_attachments_received_at ON attachments (received_at);
