from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

from app.ingestion.content_resolver import (
    InputContentResolver,
)

from app.ingestion.factory import (
    create_ingestion_pipeline,
)

from app.ingestion.fetchers.http import (
    HTTPFetcher,
)

from app.ingestion.ocr.factory import (
    create_ocr_provider,
)

from app.ingestion.ocr.provider import (
    OCRProvider,
)

from app.ingestion.speech.factory import (
    create_asr_provider,
)

from app.ingestion.speech.provider import (
    ASRProvider,
)

from app.ingestion.video.asr import (
    VideoASRService,
)

from app.ingestion.video.ffmpeg import (
    FFmpegAudioExtractor,
)

from app.ingestion.video.vision_dependencies import (
    get_vision_service,
)

from app.ingestion.video.vision_orchestration import (
    VideoVisionService,
)

from app.ingestion.results import (
    IngestionResult,
)

from app.ingestion.schemas import (
    IngestionRequest,
    InputType,
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
    - Fetch remote URL content when the source type is URL.
    - Persist the original source content.
    - Persist source metadata in PostgreSQL.
    - Resolve content for downstream processing.
    - Configure OCR enrichment.
    - Configure ASR enrichment.
    - Configure video ASR enrichment.
    - Configure video Vision enrichment.
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
              ├── OCRProvider
              │      ↑
              │   OCR Factory
              │
              ├── ASRProvider
              │      ↑
              │   ASR Factory
              │
              ├── VideoASRService
              │      ├── FFmpegAudioExtractor
              │      └── ASRProvider
              │
              └── VideoVisionService
                     ├── FFmpegFrameExtractor
                     └── VisionService
                            ↑
                         Vision Factory
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
        asr_provider: ASRProvider | None = None,
        video_asr_service: VideoASRService | None = None,
        video_vision_service: VideoVisionService | None = None,
        url_fetcher: HTTPFetcher | None = None,
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
        # ASR provider
        #
        # Production default:
        # Configured provider from ASR factory.
        #
        # Tests can inject a fake ASR provider.
        # =====================================================

        self.asr_provider = (
            asr_provider
            if asr_provider is not None
            else create_asr_provider()
        )

        # =====================================================
        # Video ASR service
        #
        # Production default:
        # FFmpeg audio extraction + configured ASR provider.
        #
        # Tests can inject a fake VideoASRService.
        # =====================================================

        self.video_asr_service = (
            video_asr_service
            if video_asr_service is not None
            else VideoASRService(
                audio_extractor=FFmpegAudioExtractor(),
                asr_provider=self.asr_provider,
            )
        )

        # =====================================================
        # Video Vision service
        #
        # Production default:
        # FFmpeg frame extraction + configured VisionService.
        #
        # Tests can inject a fake VideoVisionService.
        #
        # Vision is optional. When Vision is disabled and no
        # service is explicitly injected, keep the dependency
        # as None instead of initializing the Vision factory.
        # =====================================================

        if video_vision_service is not None:
            self.video_vision_service = video_vision_service
        elif settings.vision_enabled:
            self.video_vision_service = VideoVisionService(
                vision_service=get_vision_service(),
            )
        else:
            self.video_vision_service = None

        # =====================================================
        # URL fetcher
        #
        # Production default:
        # Secure HTTPFetcher with URL validation,
        # redirect validation, DNS security, content-type
        # validation, timeout and response-size limits.
        #
        # Tests can inject a fake HTTPFetcher.
        # =====================================================

        self.url_fetcher = (
            url_fetcher
            if url_fetcher is not None
            else HTTPFetcher()
        )

        # =====================================================
        # Ingestion pipeline
        # =====================================================

        self.pipeline = create_ingestion_pipeline(
            content_resolver=self.content_resolver,
            ocr_provider=self.ocr_provider,
            asr_provider=self.asr_provider,
            video_asr_service=self.video_asr_service,
            video_vision_service=self.video_vision_service,
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

            URL source:
                URL
                 ↓
                HTTPFetcher
                 ↓
                Fetched HTML/content
                 ↓
                Storage upload
                 ↓
                DB persistence
                 ↓
                Processing
                 ↓
                COMPLETED / FAILED

            Other sources:

                Upload
                 ↓
                DB persistence
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
        # 2. Resolve URL content before storage/pipeline
        #
        # HTMLDocumentProcessor intentionally does NOT perform
        # network access. Therefore URL fetching belongs at the
        # application/orchestration layer.
        # =====================================================

        source_request = request.model_copy(
            update={
                "source_id": source_id,
            }
        )

        if request.input_type == InputType.URL:
            if not request.url:
                raise ValueError(
                    "URL source requires a URL."
                )

            try:
                fetched = await self.url_fetcher.fetch(
                    request.url,
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to fetch URL: {exc}"
                ) from exc

            fetch_metadata = {
                "source_url": request.url,
                "final_url": fetched.final_url,
                "http_status_code": fetched.status_code,
                "fetched_content_type": fetched.content_type,
            }

            source_request = source_request.model_copy(
                update={
                    "content": fetched.content,
                    "mime_type": fetched.content_type,
                    "metadata": {
                        **request.metadata,
                        **fetch_metadata,
                    },
                }
            )

        # =====================================================
        # 3. Upload original/resolved source content
        # =====================================================

        if source_request.content is not None:
            content = (
                source_request.content
                if isinstance(
                    source_request.content,
                    bytes,
                )
                else source_request.content.encode(
                    "utf-8"
                )
            )

            storage_object = (
                await self.storage.upload_source(
                    object_id=source_id,
                    filename=(
                        source_request.filename
                        or source_request.title
                        or "source"
                    ),
                    content_type=(
                        source_request.mime_type
                        or "application/octet-stream"
                    ),
                    content=content,
                    metadata=source_request.metadata,
                )
            )

        # =====================================================
        # 4. Persist source metadata
        # =====================================================

        try:
            source = (
                await self.source_service.create_source(
                    request=source_request,
                    storage_key=(
                        storage_object.storage_key
                        if storage_object is not None
                        else None
                    ),
                    storage_uri=(
                        storage_object.uri
                        if storage_object is not None
                        else source_request.storage_uri
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
                else source_request.storage_uri
            ),
            content_hash=(
                storage_object.content_hash
                if storage_object is not None
                else None
            ),
            status=ProcessingStatus.COMPLETED,
            metadata=source_request.metadata,
        )