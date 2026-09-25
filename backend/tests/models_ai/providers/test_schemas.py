from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models_ai.providers.schemas import (
    LLMProviderConfig,
)


class TestLLMProviderConfig:
    def test_valid_configuration(self) -> None:
        config = LLMProviderConfig(
            provider="groq",
            api_key="test-api-key",
            base_url="https://api.example.com",
            timeout_seconds=30.0,
            default_model="test-model",
            metadata={
                "environment": "test",
            },
        )

        assert config.provider == "groq"
        assert config.api_key == "test-api-key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout_seconds == 30.0
        assert config.default_model == "test-model"
        assert config.metadata == {
            "environment": "test",
        }

    def test_optional_fields_default(self) -> None:
        config = LLMProviderConfig(
            provider="test-provider",
        )

        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout_seconds == 60.0
        assert config.default_model is None
        assert config.metadata == {}

    def test_provider_is_required(self) -> None:
        with pytest.raises(ValidationError):
            LLMProviderConfig()

    @pytest.mark.parametrize(
        "field",
        [
            "provider",
            "api_key",
            "base_url",
            "default_model",
        ],
    )
    def test_blank_strings_are_rejected(
        self,
        field: str,
    ) -> None:
        values = {
            "provider": "test-provider",
            "api_key": "test-key",
            "base_url": "https://example.com",
            "default_model": "test-model",
        }

        values[field] = "   "

        with pytest.raises(ValidationError):
            LLMProviderConfig(**values)

    def test_api_key_can_be_omitted(self) -> None:
        config = LLMProviderConfig(
            provider="test-provider",
        )

        assert config.api_key is None

    def test_base_url_can_be_omitted(self) -> None:
        config = LLMProviderConfig(
            provider="test-provider",
        )

        assert config.base_url is None

    def test_default_model_can_be_omitted(self) -> None:
        config = LLMProviderConfig(
            provider="test-provider",
        )

        assert config.default_model is None

    @pytest.mark.parametrize(
        "timeout_seconds",
        [
            0,
            -1,
            -0.1,
        ],
    )
    def test_timeout_must_be_positive(
        self,
        timeout_seconds: float,
    ) -> None:
        with pytest.raises(ValidationError):
            LLMProviderConfig(
                provider="test-provider",
                timeout_seconds=timeout_seconds,
            )

    def test_configuration_is_immutable(self) -> None:
        config = LLMProviderConfig(
            provider="test-provider",
        )

        with pytest.raises(ValidationError):
            config.provider = "another-provider"

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMProviderConfig(
                provider="test-provider",
                unexpected_field=True,
            )

    def test_string_values_are_normalized(self) -> None:
        config = LLMProviderConfig(
            provider="  groq  ",
            api_key="  test-key  ",
            base_url="  https://example.com  ",
            default_model="  test-model  ",
        )

        assert config.provider == "groq"
        assert config.api_key == "test-key"
        assert config.base_url == "https://example.com"
        assert config.default_model == "test-model"