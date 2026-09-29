from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.schemas import (
    IngestionRequest,
    ProcessingStatus,
)
from app.models.source import Source


class SourcePersistenceService:
    """Persist ingestion source metadata in PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_source(
        self,
        *,
        request: IngestionRequest,
        storage_key: str | None = None,
        storage_uri: str | None = None,
        content_hash: str | None = None,
        status: ProcessingStatus = ProcessingStatus.PENDING,
    ) -> Source:
        """Create a source record linked to the ingestion source ID."""

        if request.project_id is None:
            raise ValueError(
                "project_id is required to persist a source."
            )

        if request.source_id is None:
            raise ValueError(
                "source_id is required to persist a source."
            )

        source = Source(
            id=request.source_id,
            project_id=request.project_id,
            source_type=request.input_type.value,
            title=request.title,
            original_filename=request.filename,
            mime_type=request.mime_type,
            storage_key=storage_key,
            storage_uri=storage_uri,
            content_hash=content_hash,
            source_metadata=request.metadata,
            status=status.value,
        )

        self.session.add(source)

        await self.session.flush()

        return source