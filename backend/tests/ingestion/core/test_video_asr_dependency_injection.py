from unittest.mock import Mock

from app.ingestion.content_resolver import InputContentResolver
from app.ingestion.enrichment import ContentEnrichmentService
from app.ingestion.video.asr import VideoASRService


def test_video_asr_service_can_be_injected():
    resolver = Mock(spec=InputContentResolver)
    video_asr_service = Mock(spec=VideoASRService)

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr_service,
    )

    assert service.video_asr_service is video_asr_service


def test_video_asr_service_is_optional():
    resolver = Mock(spec=InputContentResolver)

    service = ContentEnrichmentService(
        content_resolver=resolver,
    )

    assert service.video_asr_service is None