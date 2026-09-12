from __future__ import annotations

from app.ingestion.content_resolver import (
    InputContentResolver,
)

from app.ingestion.detectors import (
    DefaultInputDetector,
)

from app.ingestion.enrichment import (
    ContentEnrichmentService,
)

from app.ingestion.normalization import (
    DefaultContentNormalizer,
)

from app.ingestion.ocr.provider import (
    OCRProvider,
)

from app.ingestion.speech.provider import (
    ASRProvider,
)

from app.ingestion.pipeline import (
    IngestionPipeline,
)

from app.ingestion.registry import (
    create_processor_router,
)


def create_ingestion_pipeline(
    *,
    content_resolver: InputContentResolver | None = None,
    ocr_provider: OCRProvider | None = None,
    asr_provider: ASRProvider | None = None,
) -> IngestionPipeline:
    """
    Create the production ingestion pipeline.

    Enrichment dependencies are injected so the pipeline remains
    independent of concrete storage, OCR, and ASR implementations.

    Parameters:
        content_resolver:
            Resolves source bytes for enrichment.

        ocr_provider:
            Optional OCR provider used for IMAGE inputs.

        asr_provider:
            Optional ASR provider used for AUDIO inputs.

    Architecture:

        Input
          ↓
        Detector
          ↓
        Processor Router
          ↓
        Content Processor
          ↓
        Content Enrichment
          ├── OCR
          └── ASR
          ↓
        Normalizer
          ↓
        Canonical Content
    """

    # =========================================================
    # Input detection
    # =========================================================

    detector = DefaultInputDetector()

    # =========================================================
    # Content processor routing
    # =========================================================

    router = create_processor_router()

    # =========================================================
    # Canonical content normalization
    # =========================================================

    normalizer = DefaultContentNormalizer()

    # =========================================================
    # Optional content enrichment
    # =========================================================

    enrichment = None

    if content_resolver is not None:
        enrichment = ContentEnrichmentService(
            content_resolver=content_resolver,
            ocr_provider=ocr_provider,
            asr_provider=asr_provider,
        )

    # =========================================================
    # Production ingestion pipeline
    # =========================================================

    return IngestionPipeline(
        detector=detector,
        router=router,
        normalizer=normalizer,
        enrichment=enrichment,
    )