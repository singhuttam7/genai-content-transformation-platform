from __future__ import annotations

from app.ingestion.detectors import DefaultInputDetector
from app.ingestion.normalization import DefaultContentNormalizer
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.registry import create_processor_router


def create_ingestion_pipeline() -> IngestionPipeline:
    """Create the production ingestion pipeline."""

    detector = DefaultInputDetector()
    router = create_processor_router()
    normalizer = DefaultContentNormalizer()

    return IngestionPipeline(
        detector=detector,
        router=router,
        normalizer=normalizer,
    )