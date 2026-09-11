from datetime import datetime, timedelta, timezone

from psycopg import Connection

from contracts.models import AttachmentMetadata


ATTACHMENT_HORIZON_HOURS = 48
ATTACHMENT_HORIZON = timedelta(hours=ATTACHMENT_HORIZON_HOURS)


class AttachmentAssociationError(ValueError):
    pass


class AttachmentStore:
    def persist(
        self,
        connection: Connection,
        *,
        message_id: str,
        case_id: str | None = None,
        object_key: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        content_hash: str | None = None,
        received_at: datetime | None = None,
        storage_bucket: str = "attachments",
        is_private: bool = True,
        retention_class: str = "evidence",
        metadata: AttachmentMetadata | None = None,
        now: datetime | None = None,
    ) -> bool:
        if metadata is None:
            if object_key is None or mime_type is None or size_bytes is None or content_hash is None or received_at is None:
                raise AttachmentAssociationError("attachment metadata is incomplete")
            metadata = AttachmentMetadata(
                object_key=object_key,
                mime_type=mime_type,
                size_bytes=size_bytes,
                content_hash=content_hash,
                received_at=received_at,
                storage_bucket=storage_bucket,
                is_private=is_private,
                retention_class=retention_class,
            )
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None or metadata.received_at.tzinfo is None:
            raise AttachmentAssociationError("timestamps must include a timezone")
        received_at_utc = metadata.received_at.astimezone(timezone.utc)
        current_time_utc = current_time.astimezone(timezone.utc)
        if received_at_utc > current_time_utc:
            raise AttachmentAssociationError("attachment timestamp cannot be in the future")
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT message.received_at, link.case_id
                FROM message_case_links AS link
                JOIN raw_messages AS message ON message.message_id = link.message_id
                WHERE link.message_id = %s
                """,
                (message_id,),
            )
            link = cursor.fetchone()
            if link is None:
                raise AttachmentAssociationError("message is not linked to a case")
            if case_id is not None and link[1] != case_id:
                raise AttachmentAssociationError("message is linked to a different case")
            message_received_at = link[0].astimezone(timezone.utc)
            if current_time_utc > message_received_at + ATTACHMENT_HORIZON:
                raise AttachmentAssociationError("attachment association window has expired")
            if not message_received_at <= received_at_utc <= current_time_utc:
                raise AttachmentAssociationError("attachment is outside the association horizon")
            cursor.execute(
                """
                INSERT INTO attachments (
                    message_id, object_key, mime_type, size_bytes, content_hash, received_at,
                    storage_bucket, is_private, retention_class
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (object_key) DO NOTHING
                RETURNING attachment_id
                """,
                (
                    message_id, metadata.object_key, metadata.mime_type, metadata.size_bytes,
                    metadata.content_hash.lower(), received_at_utc,
                    metadata.storage_bucket, metadata.is_private, metadata.retention_class,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is not None:
                return True

            cursor.execute(
                """
                SELECT message_id, mime_type, size_bytes, content_hash, received_at,
                       storage_bucket, is_private, retention_class
                FROM attachments
                WHERE object_key = %s
                FOR UPDATE
                """,
                (metadata.object_key,),
            )
            existing = cursor.fetchone()
            if existing is None:
                raise AttachmentAssociationError("attachment was not persisted")

            existing_received_at = existing[4].astimezone(timezone.utc)
            existing_metadata = (
                existing[0], existing[1], existing[2], existing[3].lower(),
                existing_received_at, existing[5], existing[6], existing[7],
            )
            requested_metadata = (
                message_id, metadata.mime_type, metadata.size_bytes,
                metadata.content_hash.lower(), received_at_utc,
                metadata.storage_bucket, metadata.is_private, metadata.retention_class,
            )
            if existing_metadata != requested_metadata:
                raise AttachmentAssociationError("attachment object key already has different metadata")
            return False
