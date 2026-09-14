from __future__ import annotations

import pytest

from app.core.config import settings
from app.ingestion.video.providers.local.ollama import (
    OllamaVisionRuntime,
)
from app.ingestion.video.providers.local.provider import (
    LocalVisionProvider,
)
from app.ingestion.video.vision_factory import (
    create_vision_provider,
)


def test_create_vision_provider_returns_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "local",
    )

    provider = create_vision_provider()

    assert isinstance(
        provider,
        LocalVisionProvider,
    )


def test_create_vision_provider_creates_ollama_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "local",
    )

    provider = create_vision_provider()

    assert isinstance(
        provider,
        LocalVisionProvider,
    )

    assert isinstance(
        provider.runtime,
        OllamaVisionRuntime,
    )


def test_factory_propagates_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "local",
    )

    monkeypatch.setattr(
        settings,
        "vision_model",
        "test-vision-model",
    )

    provider = create_vision_provider()

    assert isinstance(
        provider,
        LocalVisionProvider,
    )

    runtime = provider.runtime

    assert isinstance(
        runtime,
        OllamaVisionRuntime,
    )

    assert runtime.config.model == (
        "test-vision-model"
    )


def test_factory_propagates_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "local",
    )

    monkeypatch.setattr(
        settings,
        "vision_base_url",
        "http://test-ollama:11434",
    )

    monkeypatch.setattr(
        settings,
        "vision_timeout_seconds",
        45.0,
    )

    monkeypatch.setattr(
        settings,
        "vision_max_retries",
        3,
    )

    monkeypatch.setattr(
        settings,
        "vision_temperature",
        0.7,
    )

    monkeypatch.setattr(
        settings,
        "vision_keep_alive",
        "10m",
    )

    provider = create_vision_provider()

    assert isinstance(
        provider,
        LocalVisionProvider,
    )

    runtime = provider.runtime

    assert isinstance(
        runtime,
        OllamaVisionRuntime,
    )

    config = runtime.config

    assert config.base_url == (
        "http://test-ollama:11434"
    )

    assert config.timeout_seconds == 45.0
    assert config.max_retries == 3
    assert config.temperature == 0.7
    assert config.keep_alive == "10m"


def test_factory_normalizes_provider_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "  LOCAL  ",
    )

    provider = create_vision_provider()

    assert isinstance(
        provider,
        LocalVisionProvider,
    )


def test_factory_rejects_disabled_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        False,
    )

    with pytest.raises(
        ValueError,
        match="Vision is disabled",
    ):
        create_vision_provider()


def test_factory_rejects_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "unsupported-provider",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported vision provider",
    ):
        create_vision_provider()


def test_factory_does_not_create_http_client_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "vision_enabled",
        True,
    )

    monkeypatch.setattr(
        settings,
        "vision_provider",
        "local",
    )

    provider = create_vision_provider()

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