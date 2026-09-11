from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.content_resolver import (
    InputContentResolver,
)
from app.ingestion.factory import (
    create_ingestion_pipeline,
)
from app.ingestion.ocr.factory import (
    create_ocr_provider,
)
from app.ingestion.ocr.provider import (
    OCRProvider,
)
from app.ingestion.results import (
    IngestionResult,
)
from app.ingestion.schemas import (
    IngestionRequest,
    ProcessingStatus,
)
from app.ingestion.source_service import (
    SourcePersistenceService,
)
from app.ingestion.storage_resolver import (
    StorageBackedContentResolver,
)
from app.storage.compensation import (
    StorageCompensationService,
)
from app.storage.errors import (
    StorageCompensationError,
)
from app.storage.service import (
    StorageService,
)


class IngestionApplicationService:
    """
    Application-level orchestration service for source ingestion.

    Responsibilities:
    - Generate a stable source identity.
    - Persist the original source content.
    - Persist source metadata in PostgreSQL.
    - Resolve content for downstream processing.
    - Configure OCR enrichment.
    - Execute the ingestion pipeline.
    - Maintain source lifecycle state.
    - Compensate storage when database persistence fails.
    - Preserve compensation failures for recovery.

    Dependency architecture:

        StorageService
              ↓
        StorageBackedContentResolver
              ↓
        ContentEnrichmentService
              ↑
        OCRProvider
              ↑
        OCR Factory
              ↓
        IngestionPipeline
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        storage: StorageService,
        compensation: StorageCompensationService | None = None,
        content_resolver: InputContentResolver | None = None,
        ocr_provider: OCRProvider | None = None,
    ) -> None:
        self.session = session
        self.storage = storage

        # =====================================================
        # Storage compensation
        # =====================================================

        self.compensation = (
            compensation
            or StorageCompensationService(
                storage_service=storage,
            )
        )

        # =====================================================
        # Content resolver
        #
        # Production default:
        # StorageBackedContentResolver
        #
        # Tests can inject a fake resolver.
        # =====================================================

        self.content_resolver = (
            content_resolver
            or StorageBackedContentResolver(
                storage,
            )
        )

        # =====================================================
        # OCR provider
        #
        # Production default:
        # Configured provider from OCR factory.
        #
        # Tests can inject a fake OCR provider.
        # =====================================================

        self.ocr_provider = (
            ocr_provider
            if ocr_provider is not None
            else create_ocr_provider()
        )

        # =====================================================
        # Ingestion pipeline
        # =====================================================

        self.pipeline = create_ingestion_pipeline(
            content_resolver=self.content_resolver,
            ocr_provider=self.ocr_provider,
        )

        # =====================================================
        # Source persistence
        # =====================================================

        self.source_service = (
            SourcePersistenceService(
                session,
            )
        )

    async def ingest(
        self,
        *,
        request: IngestionRequest,
    ) -> IngestionResult:
        """
        Execute the complete source-ingestion lifecycle.

        Lifecycle:

            Upload
                ↓
            DB persistence
                ↓
            Commit
                ↓
            Processing
                ↓
            COMPLETED / FAILED

        Database persistence failure:

            Upload
                ↓
            DB failure
                ↓
            Rollback
                ↓
            Compensation
        """

        # =====================================================
        # 1. Generate stable source identity
        # =====================================================

        source_id = (
            request.source_id
            or uuid4()
        )

        storage_object = None

        # =====================================================
        # 2. Upload original source content
        # =====================================================

        if request.content is not None:
            content = (
                request.content
                if isinstance(
                    request.content,
                    bytes,
                )
                else request.content.encode(
                    "utf-8"
                )
            )

            storage_object = (
                await self.storage.upload_source(
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
            )

        # =====================================================
        # 3. Prepare source request
        # =====================================================

        source_request = request.model_copy(
            update={
                "source_id": source_id,
            }
        )

        # =====================================================
        # 4. Persist source metadata
        # =====================================================

        try:
            source = (
                await self.source_service.create_source(
                    request=source_request,
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
                    status=(
                        ProcessingStatus.PROCESSING
                    ),
                )
            )

            await self.session.commit()

        except Exception as original_error:

            # -------------------------------------------------
            # Roll back PostgreSQL transaction.
            # -------------------------------------------------

            await self.session.rollback()

            # -------------------------------------------------
            # Compensate successful storage upload.
            # -------------------------------------------------

            if storage_object is not None:
                try:
                    await (
                        self.compensation.compensate_upload(
                            storage_object.storage_key,
                        )
                    )

                except Exception as compensation_error:

                    # -------------------------------------------------
                    # Both failures are important.
                    #
                    # The database failure caused the compensation,
                    # while the compensation failure means cleanup
                    # could not be completed.
                    # -------------------------------------------------

                    raise StorageCompensationError(
                        storage_key=(
                            storage_object.storage_key
                        ),
                        original_error=original_error,
                        compensation_error=(
                            compensation_error
                        ),
                    ) from original_error

            # -------------------------------------------------
            # No storage object existed, so there is nothing
            # to compensate.
            # -------------------------------------------------

            raise

        # =====================================================
        # 5. Run content-processing pipeline
        # =====================================================

        try:
            canonical_content = (
                await self.pipeline.run(
                    source_request,
                )
            )

        except Exception:

            # -------------------------------------------------
            # The original source is intentionally retained.
            #
            # This allows:
            #
            # - retry
            # - debugging
            # - reprocessing
            # - provenance
            # - audit
            # -------------------------------------------------

            source.status = (
                ProcessingStatus.FAILED.value
            )

            await self.session.commit()

            raise

        # =====================================================
        # 6. Processing succeeded
        # =====================================================

        source.status = (
            ProcessingStatus.COMPLETED.value
        )

        await self.session.commit()

        # =====================================================
        # 7. Return unified ingestion result
        # =====================================================

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
            status=ProcessingStatus.COMPLETED,
            metadata=request.metadata,
        )