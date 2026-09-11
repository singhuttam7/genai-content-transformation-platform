from __future__ import annotations

from app.ingestion.registry import create_processor_router
from app.ingestion.schemas import InputType


def test_processor_router_supports_audio():
    router = create_processor_router()

    assert router.supports(InputType.AUDIO)


def test_processor_router_returns_audio_processor():
    router = create_processor_router()

    processor = router.get_processor(
        InputType.AUDIO,
    )

    assert processor.__class__.__name__ == (
        "AudioDocumentProcessor"
    )


def test_audio_is_registered_exactly_once():
    router = create_processor_router()

    supported_types = router.supported_types

    assert supported_types.count(InputType.AUDIO) == 1
    