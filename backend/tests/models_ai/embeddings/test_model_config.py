from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings import EmbeddingModelConfig


def test_model_config_accepts_valid_configuration() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        expected_dimension=384,
        normalized=True,
        batch_size=32,
        options={
            "device": "cpu",
        },
    )

    assert config.provider == "local"
    assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.expected_dimension == 384
    assert config.normalized is True
    assert config.batch_size == 32
    assert config.options == {"device": "cpu"}


def test_model_config_has_expected_defaults() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="test-model",
    )

    assert config.expected_dimension is None
    assert config.normalized is False
    assert config.batch_size == 32
    assert config.options == {}


def test_model_config_allows_unknown_dimension() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="test-model",
    )

    assert config.expected_dimension is None


def test_model_config_rejects_empty_provider() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="",
            model_name="test-model",
        )


def test_model_config_rejects_whitespace_provider() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="   ",
            model_name="test-model",
        )


def test_model_config_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="",
        )


def test_model_config_rejects_whitespace_model_name() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="   ",
        )


def test_model_config_strips_provider_whitespace() -> None:
    config = EmbeddingModelConfig(
        provider="  local  ",
        model_name="test-model",
    )

    assert config.provider == "local"


def test_model_config_strips_model_name_whitespace() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="  test-model  ",
    )

    assert config.model_name == "test-model"


def test_model_config_rejects_zero_dimension() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="test-model",
            expected_dimension=0,
        )


def test_model_config_rejects_negative_dimension() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="test-model",
            expected_dimension=-1,
        )


def test_model_config_rejects_zero_batch_size() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="test-model",
            batch_size=0,
        )


def test_model_config_rejects_negative_batch_size() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="test-model",
            batch_size=-1,
        )


def test_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelConfig(
            provider="local",
            model_name="test-model",
            unsupported_option=True,
        )


def test_model_config_is_immutable() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="test-model",
    )

    with pytest.raises(ValidationError):
        config.batch_size = 64  # type: ignore[misc]


def test_model_config_preserves_nested_options() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="test-model",
        options={
            "device": "cpu",
            "normalize_embeddings": True,
            "nested": {
                "value": 123,
            },
        },
    )

    assert config.options["device"] == "cpu"
    assert config.options["normalize_embeddings"] is True
    assert config.options["nested"]["value"] == 123


def test_model_config_serializes_deterministically() -> None:
    config = EmbeddingModelConfig(
        provider="local",
        model_name="test-model",
        expected_dimension=384,
        normalized=True,
        batch_size=16,
        options={
            "device": "cpu",
        },
    )

    first = config.model_dump(mode="json")
    second = config.model_dump(mode="json")

    assert first == second


def test_model_config_json_serialization_is_provider_independent() -> None:
    config = EmbeddingModelConfig(
        provider="custom-provider",
        model_name="custom-model",
        expected_dimension=768,
        batch_size=8,
        options={
            "custom_setting": "value",
        },
    )

    serialized = config.model_dump(mode="json")

    assert serialized == {
        "provider": "custom-provider",
        "model_name": "custom-model",
        "expected_dimension": 768,
        "normalized": False,
        "batch_size": 8,
        "options": {
            "custom_setting": "value",
        },
    }