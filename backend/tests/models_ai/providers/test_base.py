from __future__ import annotations

import pytest
from pydantic import ValidationError
from app.models_ai.llm.errors import LLMProviderError
from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
)
from app.models_ai.providers.base import (
    LLMProviderAdapter,
)
from app.models_ai.providers.schemas import (
    LLMProviderConfig,
)


def build_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        provider="test-provider",
        api_key="test-key",
        default_model="test-model",
    )


def build_request(
    provider: str = "test-provider",
) -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Explain the architecture.",
            ),
        ],
        model=LLMModelInfo(
            provider=provider,
            model_name="test-model",
        ),
    )


def build_response() -> LLMResponse:
    return LLMResponse(
        text="Generated response.",
        model=LLMModelInfo(
            provider="test-provider",
            model_name="test-model",
        ),
    )


class FakeProviderAdapter(LLMProviderAdapter):
    """Test implementation of the provider adapter foundation."""

    def __init__(
        self,
        config: LLMProviderConfig,
    ) -> None:
        super().__init__(config)
        self.received_request: LLMRequest | None = None

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.validate_request(request)

        self.received_request = request

        return build_response()


class TestLLMProviderAdapter:
    def test_adapter_is_constructed_with_config(self) -> None:
        adapter = FakeProviderAdapter(
            build_config(),
        )

        assert adapter.config.provider == "test-provider"
        assert adapter.provider == "test-provider"

    def test_config_property_returns_same_config(self) -> None:
        config = build_config()
        adapter = FakeProviderAdapter(config)

        assert adapter.config is config

    def test_provider_property_matches_config(self) -> None:
        config = build_config()
        adapter = FakeProviderAdapter(config)

        assert adapter.provider == config.provider

    def test_matching_provider_is_accepted(self) -> None:
        adapter = FakeProviderAdapter(
            build_config(),
        )

        adapter.validate_request(
            build_request("test-provider"),
        )

    def test_mismatched_provider_is_rejected(self) -> None:
        adapter = FakeProviderAdapter(
            build_config(),
        )

        with pytest.raises(LLMProviderError) as exc_info:
            adapter.validate_request(
                build_request("different-provider"),
            )

        assert str(exc_info.value) == (
            "LLM request provider does not match "
            "the configured adapter provider."
        )
        assert exc_info.value.provider == "test-provider"
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_concrete_adapter_can_generate(self) -> None:
        adapter = FakeProviderAdapter(
            build_config(),
        )
        request = build_request()

        response = await adapter.generate(request)

        assert isinstance(response, LLMResponse)
        assert response.text == "Generated response."
        assert adapter.received_request is request

    def test_adapter_implements_llm_port(self) -> None:
        adapter = FakeProviderAdapter(
            build_config(),
        )

        from app.models_ai.llm.port import LLMPort

        assert isinstance(adapter, LLMPort)

    def test_adapter_requires_generate_implementation(self) -> None:
        with pytest.raises(TypeError):

            class IncompleteAdapter(
                LLMProviderAdapter,
            ):
                pass

            IncompleteAdapter(build_config())

    def test_provider_config_is_immutable(self) -> None:
        adapter = FakeProviderAdapter(
            build_config(),
        )

        with pytest.raises(ValidationError):
            adapter.config.provider = "another-provider"