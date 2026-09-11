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
) -> IngestionPipeline:
    """
    Create the production ingestion pipeline.

    Enrichment dependencies are injected so the pipeline remains
    independent of concrete storage and OCR implementations.
    """

    detector = DefaultInputDetector()

    router = create_processor_router()

    normalizer = DefaultContentNormalizer()

    enrichment = None

    if content_resolver is not None:
        enrichment = ContentEnrichmentService(
            content_resolver=content_resolver,
            ocr_provider=ocr_provider,
        )

    return IngestionPipeline(
        detector=detector,
        router=router,
        normalizer=normalizer,
        enrichment=enrichment,
    )