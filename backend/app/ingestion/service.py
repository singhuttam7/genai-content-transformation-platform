from __future__ import annotations

from uuid import UUID

from app.ingestion.schemas import (
    IngestionRequest,
)
from app.storage.schemas import StorageObject
from app.storage.service import StorageService


class IngestionService:
    """Application-level service for content ingestion.

    Coordinates ingestion-related operations while keeping
    storage implementation details outside the ingestion layer.
    """

    def __init__(
        self,
        storage: StorageService,
    ) -> None:
        self.storage = storage

    async def store_source(
        self,
        *,
        request: IngestionRequest,
        object_id: UUID,
    ) -> StorageObject:
        """Store the original source content."""

        if request.content is None:
            raise ValueError(
                "Source content is required for storage."
            )

        filename = (
            request.filename
            or request.title
            or "source"
        )

        content_type = (
            request.mime_type
            or "application/octet-stream"
        )

        return await self.storage.upload_source(
            object_id=object_id,
            filename=filename,
            content_type=content_type,
            content=request.content,
            metadata=request.metadata,
        )