from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from app.models_ai.gateway import LLMGateway
from app.models_ai.llm.errors import (
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from app.models_ai.llm.port import LLMPort
from app.models_ai.llm.resilience import LLMRetryPolicy
from app.models_ai.llm.resilient_port import ResilientLLMPort
from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
)


def build_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Explain RAG.",
            ),
        ],
        model=LLMModelInfo(
            provider="test-provider",
            model_name="test-model",
        ),
    )


def build_response() -> LLMResponse:
    return LLMResponse(
        text="RAG retrieves relevant knowledge before generation.",
        model=LLMModelInfo(
            provider="test-provider",
            model_name="test-model",
        ),
    )


class SummaryOutput(BaseModel):
    title: str
    summary: str


class FakeLLMPort(LLMPort):
    """Simple fake port used to test the gateway contract."""

    def __init__(
        self,
        response: object,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.received_request: LLMRequest | None = None

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.received_request = request

        if self.error is not None:
            raise self.error

        return self.response  # type: ignore[return-value]


class TestLLMGateway:
    @pytest.mark.asyncio
    async def test_gateway_returns_port_response(self) -> None:
        expected_response = build_response()
        port = FakeLLMPort(expected_response)
        gateway = LLMGateway(port)

        response = await gateway.generate(
            build_request(),
        )

        assert response is expected_response

    @pytest.mark.asyncio
    async def test_gateway_passes_request_to_port(self) -> None:
        request = build_request()
        port = FakeLLMPort(build_response())
        gateway = LLMGateway(port)

        await gateway.generate(request)

        assert port.received_request is request

    @pytest.mark.asyncio
    async def test_provider_error_is_propagated(self) -> None:
        error = LLMRateLimitError(
            "Rate limit exceeded.",
            provider="test-provider",
        )
        port = FakeLLMPort(
            response=build_response(),
            error=error,
        )
        gateway = LLMGateway(port)

        with pytest.raises(LLMRateLimitError) as exc_info:
            await gateway.generate(build_request())

        assert exc_info.value is error
        assert exc_info.value.provider == "test-provider"
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_invalid_response_type_is_rejected(self) -> None:
        port = FakeLLMPort(
            response="invalid response",
        )
        gateway = LLMGateway(port)

        with pytest.raises(LLMResponseError) as exc_info:
            await gateway.generate(build_request())

        assert str(exc_info.value) == (
            "LLM provider returned an invalid response type."
        )
        assert exc_info.value.provider == "test-provider"
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_none_response_is_rejected(self) -> None:
        port = FakeLLMPort(
            response=None,
        )
        gateway = LLMGateway(port)

        with pytest.raises(LLMResponseError):
            await gateway.generate(build_request())

    @pytest.mark.asyncio
    async def test_gateway_does_not_replace_provider_errors(self) -> None:
        error = LLMRateLimitError(
            "Provider rate limit.",
            provider="groq",
        )
        port = FakeLLMPort(
            response=build_response(),
            error=error,
        )
        gateway = LLMGateway(port)

        with pytest.raises(LLMRateLimitError) as exc_info:
            await gateway.generate(build_request())

        assert exc_info.value is error

    @pytest.mark.asyncio
    async def test_generate_structured_returns_typed_model(
        self,
    ) -> None:
        response = LLMResponse(
            text=(
                '{"title":"RAG",'
                '"summary":"Retrieval augmented generation."}'
            ),
            model=build_request().model,
        )
        port = FakeLLMPort(response)
        gateway = LLMGateway(port)

        result = await gateway.generate_structured(
            build_request(),
            SummaryOutput,
        )

        assert isinstance(result, SummaryOutput)
        assert result.title == "RAG"
        assert result.summary == (
            "Retrieval augmented generation."
        )

    @pytest.mark.asyncio
    async def test_generate_structured_rejects_invalid_json(
        self,
    ) -> None:
        response = LLMResponse(
            text='{"title":"RAG"',
            model=build_request().model,
        )
        port = FakeLLMPort(response)
        gateway = LLMGateway(port)

        with pytest.raises(
            LLMStructuredOutputError,
        ) as exc_info:
            await gateway.generate_structured(
                build_request(),
                SummaryOutput,
            )

        assert str(exc_info.value) == (
            "LLM response is not valid JSON."
        )
        assert exc_info.value.provider == "test-provider"
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_generate_structured_rejects_schema_mismatch(
        self,
    ) -> None:
        response = LLMResponse(
            text='{"title":"RAG"}',
            model=build_request().model,
        )
        port = FakeLLMPort(response)
        gateway = LLMGateway(port)

        with pytest.raises(
            LLMStructuredOutputError,
        ):
            await gateway.generate_structured(
                build_request(),
                SummaryOutput,
            )

    @pytest.mark.asyncio
    async def test_generate_structured_propagates_provider_error(
        self,
    ) -> None:
        error = LLMRateLimitError(
            "Provider rate limit.",
            provider="groq",
        )
        port = FakeLLMPort(
            response=build_response(),
            error=error,
        )
        gateway = LLMGateway(port)

        with pytest.raises(LLMRateLimitError) as exc_info:
            await gateway.generate_structured(
                build_request(),
                SummaryOutput,
            )

        assert exc_info.value is error


class TestLLMGatewayWithResilience:
    @pytest.mark.asyncio
    async def test_gateway_works_through_resilient_port(self) -> None:
        provider = Mock(spec=LLMPort)

        provider.generate = AsyncMock(
            return_value=build_response(),
        )

        resilient_port = ResilientLLMPort(
            provider,
        )

        gateway = LLMGateway(resilient_port)

        response = await gateway.generate(
            build_request(),
        )

        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )
        provider.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gateway_retries_retryable_provider_failure(
        self,
    ) -> None:
        provider = Mock(spec=LLMPort)

        provider.generate = AsyncMock(
            side_effect=[
                LLMProviderError(
                    "Temporary failure.",
                    provider="test-provider",
                    retryable=True,
                ),
                build_response(),
            ],
        )

        sleep = AsyncMock()

        resilient_port = ResilientLLMPort(
            provider,
            retry_policy=LLMRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.5,
            ),
            sleep=sleep,
        )

        gateway = LLMGateway(resilient_port)

        response = await gateway.generate(
            build_request(),
        )

        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )
        assert provider.generate.await_count == 2
        sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_gateway_retries_timeout_then_succeeds(
        self,
    ) -> None:
        provider = Mock(spec=LLMPort)

        provider.generate = AsyncMock(
            side_effect=[
                LLMTimeoutError(
                    "Request timed out.",
                    provider="test-provider",
                ),
                build_response(),
            ],
        )

        sleep = AsyncMock()

        resilient_port = ResilientLLMPort(
            provider,
            retry_policy=LLMRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.5,
            ),
            sleep=sleep,
        )

        gateway = LLMGateway(resilient_port)

        response = await gateway.generate(
            build_request(),
        )

        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )
        assert provider.generate.await_count == 2
        sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_gateway_propagates_exhausted_retryable_failure(
        self,
    ) -> None:
        provider = Mock(spec=LLMPort)

        error = LLMProviderError(
            "Provider unavailable.",
            provider="test-provider",
            retryable=True,
        )

        provider.generate = AsyncMock(
            side_effect=error,
        )

        sleep = AsyncMock()

        resilient_port = ResilientLLMPort(
            provider,
            retry_policy=LLMRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.5,
            ),
            sleep=sleep,
        )

        gateway = LLMGateway(resilient_port)

        with pytest.raises(LLMProviderError) as exc_info:
            await gateway.generate(
                build_request(),
            )

        assert exc_info.value is error
        assert provider.generate.await_count == 2
        sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_gateway_propagates_timeout_from_resilient_port(
        self,
    ) -> None:
        provider = Mock(spec=LLMPort)

        error = LLMTimeoutError(
            "Request timed out.",
            provider="test-provider",
        )

        provider.generate = AsyncMock(
            side_effect=error,
        )

        resilient_port = ResilientLLMPort(
            provider,
            retry_policy=LLMRetryPolicy(
                max_attempts=1,
            ),
        )

        gateway = LLMGateway(resilient_port)

        with pytest.raises(LLMTimeoutError) as exc_info:
            await gateway.generate(
                build_request(),
            )

        assert exc_info.value is error
        assert exc_info.value.provider == "test-provider"
        assert exc_info.value.retryable is True
        provider.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gateway_structured_output_works_through_resilience(
        self,
    ) -> None:
        provider = Mock(spec=LLMPort)

        provider.generate = AsyncMock(
            return_value=LLMResponse(
                text=(
                    '{"title":"RAG",'
                    '"summary":"Retrieval augmented generation."}'
                ),
                model=build_request().model,
            ),
        )

        resilient_port = ResilientLLMPort(
            provider,
            retry_policy=LLMRetryPolicy(
                max_attempts=2,
            ),
        )

        gateway = LLMGateway(resilient_port)

        result = await gateway.generate_structured(
            build_request(),
            SummaryOutput,
        )

        assert isinstance(result, SummaryOutput)
        assert result.title == "RAG"
        assert result.summary == (
            "Retrieval augmented generation."
        )
        provider.generate.assert_awaited_once()


def test_gateway_is_exported_from_models_ai() -> None:
    from app.models_ai import LLMGateway as ExportedLLMGateway

    assert ExportedLLMGateway is LLMGateway