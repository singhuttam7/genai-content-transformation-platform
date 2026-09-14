from __future__ import annotations

from app.ingestion.enrichment.service import (
    ContentEnrichmentService,
)
from app.ingestion.factory import (
    create_ingestion_pipeline,
)
from app.ingestion.pipeline import (
    IngestionPipeline,
)


class FakeContentResolver:
    """
    Minimal dependency used to enable enrichment creation.
    """

    async def resolve(self, request):
        return b"test-content"


class FakeVideoVisionService:
    """
    Minimal video vision dependency used to verify
    dependency injection at the factory boundary.
    """

    async def analyze(
        self,
        media: bytes,
        *,
        video_info,
        vision_request=None,
        frame_extraction_request=None,
    ):
        raise NotImplementedError


def test_factory_returns_ingestion_pipeline() -> None:
    resolver = FakeContentResolver()

    pipeline = create_ingestion_pipeline(
        content_resolver=resolver,
    )

    assert isinstance(
        pipeline,
        IngestionPipeline,
    )


def test_factory_creates_enrichment_when_resolver_is_provided() -> None:
    resolver = FakeContentResolver()

    pipeline = create_ingestion_pipeline(
        content_resolver=resolver,
    )

    assert pipeline.enrichment is not None

    assert isinstance(
        pipeline.enrichment,
        ContentEnrichmentService,
    )


def test_factory_injects_video_vision_service() -> None:
    resolver = FakeContentResolver()
    video_vision_service = FakeVideoVisionService()

    pipeline = create_ingestion_pipeline(
        content_resolver=resolver,
        video_vision_service=video_vision_service,
    )

    assert pipeline.enrichment is not None

    assert (
        pipeline.enrichment.video_vision_service
        is video_vision_service
    )


def test_factory_preserves_other_optional_dependencies() -> None:
    resolver = FakeContentResolver()
    video_vision_service = FakeVideoVisionService()

    pipeline = create_ingestion_pipeline(
        content_resolver=resolver,
        video_vision_service=video_vision_service,
    )

    enrichment = pipeline.enrichment

    assert enrichment is not None

    assert (
        enrichment.content_resolver
        is resolver
    )

    assert (
        enrichment.video_vision_service
        is video_vision_service
    )


def test_factory_without_resolver_has_no_enrichment() -> None:
    pipeline = create_ingestion_pipeline()

    assert pipeline.enrichment is None