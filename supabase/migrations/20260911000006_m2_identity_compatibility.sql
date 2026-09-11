ALTER TABLE raw_messages DROP CONSTRAINT IF EXISTS uq_raw_messages_source;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_raw_messages_source_identity'
    ) THEN
        ALTER TABLE raw_messages
            ADD CONSTRAINT uq_raw_messages_source_identity
            UNIQUE (tenant_id, connector_id, account_id, source_message_id);
    END IF;
END $$;
