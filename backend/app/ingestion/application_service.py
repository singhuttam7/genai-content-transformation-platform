from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.factory import create_ingestion_pipeline
from app.ingestion.results import IngestionResult
from app.ingestion.schemas import IngestionRequest
from app.ingestion.source_service import SourcePersistenceService
from app.storage.service import StorageService


class IngestionApplicationService:
    """Application-level orchestration for source ingestion."""

    def __init__(
        self,
        session: AsyncSession,
        storage: StorageService,
    ) -> None:
        self.session = session
        self.storage = storage
        self.pipeline = create_ingestion_pipeline()
        self.source_service = SourcePersistenceService(session)

    async def ingest(
        self,
        *,
        request: IngestionRequest,
    ) -> IngestionResult:
        """Store, persist, process, and normalize a source."""

        source_id = request.source_id or uuid4()

        storage_object = None

        # ---------------------------------------------------------
        # 1. Store original source
        # ---------------------------------------------------------
        if request.content is not None:
            content = (
                request.content
                if isinstance(request.content, bytes)
                else request.content.encode("utf-8")
            )

            storage_object = await self.storage.upload_source(
                object_id=source_id,
                filename=(
                    request.filename
                    or request.title
                    or "source"
                ),
                content_type=(
                    request.mime_type
                    or "application/octet-stream"
                ),
                content=content,
                metadata=request.metadata,
            )

        # ---------------------------------------------------------
        # 2. Persist source metadata
        # ---------------------------------------------------------
        source = await self.source_service.create_source(
            request=request.model_copy(
                update={"source_id": source_id}
            ),
            storage_uri=(
                storage_object.uri
                if storage_object is not None
                else request.storage_uri
            ),
            content_hash=(
                storage_object.content_hash
                if storage_object is not None
                else None
            ),
            status="PROCESSING",
        )

        # ---------------------------------------------------------
        # 3. Run ingestion pipeline
        # ---------------------------------------------------------
        try:
            canonical_content = await self.pipeline.run(
                request.model_copy(
                    update={"source_id": source_id}
                )
            )

            source.status = "COMPLETED"

            await self.session.commit()

            return IngestionResult(
                source_id=source_id,
                canonical_content=canonical_content,
                storage_key=(
                    storage_object.storage_key
                    if storage_object is not None
                    else None
                ),
                storage_uri=(
                    storage_object.uri
                    if storage_object is not None
                    else request.storage_uri
                ),
                content_hash=(
                    storage_object.content_hash
                    if storage_object is not None
                    else None
                ),
                status="COMPLETED",
                metadata=request.metadata,
            )

        except Exception:
            source.status = "FAILED"

            await self.session.commit()

            raise