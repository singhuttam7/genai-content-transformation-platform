from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.models_ai.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
)
from app.models_ai.providers.groq import GroqLLMAdapter
from app.models_ai.providers.schemas import LLMProviderConfig


def build_config(
    api_key: str | None = "test-api-key",
) -> LLMProviderConfig:
    return LLMProviderConfig(
        provider="groq",
        api_key=api_key,
        default_model="openai/gpt-oss-120b",
    )


def build_request(
    provider: str = "groq",
) -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            LLMMessage(
                role="user",
                content="Explain RAG.",
            ),
        ],
        model=LLMModelInfo(
            provider=provider,
            model_name="openai/gpt-oss-120b",
        ),
        temperature=0.3,
        max_tokens=200,
    )


def build_completion(
    *,
    text: str = "RAG retrieves relevant knowledge before generation.",
    finish_reason: str = "stop",
    usage: object | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=text,
    )

    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        id="test-completion-id",
        choices=[choice],
        usage=usage,
    )


class FakeCompletions:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return self.response


class FakeChat:
    def __init__(
        self,
        completions: FakeCompletions,
    ) -> None:
        self.completions = completions


class FakeGroqClient:
    def __init__(
        self,
        completions: FakeCompletions,
    ) -> None:
        self.chat = FakeChat(completions)


class TestGroqLLMAdapter:
    def test_adapter_requires_groq_provider(self) -> None:
        config = LLMProviderConfig(
            provider="gemini",
            api_key="test-key",
        )

        with pytest.raises(LLMConfigurationError):
            GroqLLMAdapter(config)

    def test_adapter_requires_api_key_without_client(self) -> None:
        with pytest.raises(LLMConfigurationError) as exc_info:
            GroqLLMAdapter(
                build_config(api_key=None),
            )

        assert exc_info.value.provider == "groq"

    def test_injected_client_allows_missing_api_key(self) -> None:
        completions = FakeCompletions(
            response=build_completion(),
        )
        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(api_key=None),
            client=client,  # type: ignore[arg-type]
        )

        assert adapter.client is client

    @pytest.mark.asyncio
    async def test_generate_maps_request_to_groq(self) -> None:
        completions = FakeCompletions(
            response=build_completion(),
        )
        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        request = build_request()

        response = await adapter.generate(request)

        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )

        assert len(completions.calls) == 1

        call = completions.calls[0]

        assert call["model"] == "openai/gpt-oss-120b"
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 200
        assert call["messages"] == [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Explain RAG.",
            },
        ]

    @pytest.mark.asyncio
    async def test_generate_maps_response(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        completions = FakeCompletions(
            response=build_completion(
                usage=usage,
            ),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        response = await adapter.generate(
            build_request(),
        )

        assert response.text == (
            "RAG retrieves relevant knowledge before generation."
        )

        assert response.model.provider == "groq"
        assert response.model.model_name == (
            "openai/gpt-oss-120b"
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens == 50
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 70

        assert response.finish_reason == "stop"

        assert response.metadata["provider"] == "groq"
        assert response.metadata["completion_id"] == (
            "test-completion-id"
        )

    @pytest.mark.asyncio
    async def test_generate_without_usage(self) -> None:
        completions = FakeCompletions(
            response=build_completion(
                usage=None,
            ),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        response = await adapter.generate(
            build_request(),
        )

        assert response.usage is None

    @pytest.mark.asyncio
    async def test_mismatched_provider_is_rejected(self) -> None:
        completions = FakeCompletions(
            response=build_completion(),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMProviderError):
            await adapter.generate(
                build_request(provider="gemini"),
            )

        assert completions.calls == []

    @pytest.mark.asyncio
    async def test_empty_choices_raise_response_error(self) -> None:
        completion = SimpleNamespace(
            id="test-id",
            choices=[],
            usage=None,
        )

        completions = FakeCompletions(
            response=completion,
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMResponseError):
            await adapter.generate(
                build_request(),
            )

    @pytest.mark.asyncio
    async def test_empty_response_text_raises_response_error(self) -> None:
        completions = FakeCompletions(
            response=build_completion(
                text="   ",
            ),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMResponseError):
            await adapter.generate(
                build_request(),
            )

    @pytest.mark.asyncio
    async def test_authentication_error_is_translated(self) -> None:
        error = Exception("Unauthorized")
        error.status_code = 401  # type: ignore[attr-defined]

        completions = FakeCompletions(
            error=error,
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await adapter.generate(
                build_request(),
            )

        assert exc_info.value.provider == "groq"
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_translated(self) -> None:
        error = Exception("Rate limited")
        error.status_code = 429  # type: ignore[attr-defined]

        completions = FakeCompletions(
            error=error,
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMRateLimitError) as exc_info:
            await adapter.generate(
                build_request(),
            )

        assert exc_info.value.provider == "groq"
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_server_error_is_translated_as_retryable(self) -> None:
        error = Exception("Server error")
        error.status_code = 500  # type: ignore[attr-defined]

        completions = FakeCompletions(
            error=error,
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMProviderError) as exc_info:
            await adapter.generate(
                build_request(),
            )

        assert exc_info.value.provider == "groq"
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_timeout_error_is_translated(self) -> None:
        completions = FakeCompletions(
            error=TimeoutError("Request timed out"),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMTimeoutError) as exc_info:
            await adapter.generate(
                build_request(),
            )

        assert exc_info.value.provider == "groq"
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_value_error_is_translated_as_configuration_error(
        self,
    ) -> None:
        completions = FakeCompletions(
            error=ValueError("Invalid request"),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMConfigurationError) as exc_info:
            await adapter.generate(
                build_request(),
            )

        assert exc_info.value.provider == "groq"

    @pytest.mark.asyncio
    async def test_unknown_provider_error_is_translated(self) -> None:
        completions = FakeCompletions(
            error=RuntimeError("Unexpected provider failure"),
        )

        client = FakeGroqClient(completions)

        adapter = GroqLLMAdapter(
            build_config(),
            client=client,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMProviderError) as exc_info:
            await adapter.generate(
                build_request(),
            )

        assert str(exc_info.value) == (
            "Unexpected provider failure"
        )
        assert exc_info.value.provider == "groq"
        assert exc_info.value.retryable is False

    def test_build_messages_preserves_order(self) -> None:
        messages = [
            LLMMessage(
                role="system",
                content="System message.",
            ),
            LLMMessage(
                role="user",
                content="User message.",
            ),
            LLMMessage(
                role="assistant",
                content="Assistant message.",
            ),
        ]

        result = GroqLLMAdapter._build_messages(
            messages,
        )

        assert result == [
            {
                "role": "system",
                "content": "System message.",
            },
            {
                "role": "user",
                "content": "User message.",
            },
            {
                "role": "assistant",
                "content": "Assistant message.",
            },
        ]