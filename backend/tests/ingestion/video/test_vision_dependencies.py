from __future__ import annotations

import pytest

from app.ingestion.video.providers.local.ollama import (
    OllamaVisionRuntime,
)
from app.ingestion.video.providers.local.provider import (
    LocalVisionProvider,
)
from app.ingestion.video.vision import VisionService
from app.ingestion.video.vision_dependencies import (
    get_vision_service,
)


@pytest.fixture(autouse=True)
def clear_vision_service_cache() -> None:
    """
    Clear the cached dependency before each test.

    This prevents one test's service instance from affecting
    another test.
    """

    get_vision_service.cache_clear()


def test_get_vision_service_returns_vision_service() -> None:
    service = get_vision_service()

    assert isinstance(
        service,
        VisionService,
    )


def test_get_vision_service_uses_local_provider() -> None:
    service = get_vision_service()

    assert isinstance(
        service.provider,
        LocalVisionProvider,
    )


def test_get_vision_service_uses_ollama_runtime() -> None:
    service = get_vision_service()

    provider = service.provider

    assert isinstance(
        provider,
        LocalVisionProvider,
    )

    assert isinstance(
        provider.runtime,
        OllamaVisionRuntime,
    )


def test_get_vision_service_returns_cached_instance() -> None:
    first = get_vision_service()
    second = get_vision_service()

    assert first is second


def test_cached_service_reuses_provider() -> None:
    first = get_vision_service()
    second = get_vision_service()

    assert first.provider is second.provider


def test_cached_service_reuses_runtime() -> None:
    first = get_vision_service()
    second = get_vision_service()

    first_provider = first.provider
    second_provider = second.provider

    assert isinstance(
        first_provider,
        LocalVisionProvider,
    )

    assert isinstance(
        second_provider,
        LocalVisionProvider,
    )

    assert first_provider.runtime is second_provider.runtime


def test_dependency_does_not_create_http_client() -> None:
    service = get_vision_service()

    provider = service.provider

    assert isinstance(
        provider,
        LocalVisionProvider,
    )

    runtime = provider.runtime

    assert isinstance(
        runtime,
        OllamaVisionRuntime,
    )

    assert runtime.client is None


def test_cache_can_be_cleared_and_new_service_created() -> None:
    first = get_vision_service()

    get_vision_service.cache_clear()

    second = get_vision_service()

    assert first is not second
    assert first.provider is not second.provider