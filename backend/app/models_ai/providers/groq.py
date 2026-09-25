from __future__ import annotations

from typing import Any

from groq import AsyncGroq

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
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from app.models_ai.providers.base import LLMProviderAdapter
from app.models_ai.providers.schemas import LLMProviderConfig


class GroqLLMAdapter(LLMProviderAdapter):
    """
    Groq implementation of the provider-independent LLMPort.

    The adapter translates the platform's LLM contracts into Groq's
    chat-completions API and translates provider failures back into
    provider-independent LLM errors.
    """

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        client: AsyncGroq | None = None,
    ) -> None:
        if config.provider != "groq":
            raise LLMConfigurationError(
                "Groq adapter requires provider='groq'.",
                provider="groq",
            )

        if client is None and config.api_key is None:
            raise LLMConfigurationError(
                "Groq API key is required when no client is supplied.",
                provider="groq",
            )

        super().__init__(config)

        self._client = client or AsyncGroq(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    @property
    def client(self) -> AsyncGroq:
        """Return the configured Groq client."""

        return self._client

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response using Groq chat completions.
        """

        self.validate_request(request)

        try:
            completion = await self._client.chat.completions.create(
                model=request.model.model_name,
                messages=self._build_messages(request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            raise self._translate_exception(exc) from exc

        return self._build_response(
            request=request,
            completion=completion,
        )

    @staticmethod
    def _build_messages(
        messages: list[LLMMessage],
    ) -> list[dict[str, str]]:
        """Convert platform messages into Groq chat messages."""

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    def _build_response(
        self,
        request: LLMRequest,
        completion: Any,
    ) -> LLMResponse:
        """Convert a Groq completion into the platform response contract."""

        if not completion.choices:
            raise LLMResponseError(
                "Groq returned no completion choices.",
                provider="groq",
            )

        choice = completion.choices[0]

        if choice.message is None:
            raise LLMResponseError(
                "Groq returned a completion without a message.",
                provider="groq",
            )

        text = choice.message.content

        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(
                "Groq returned an empty response.",
                provider="groq",
            )

        usage = self._build_usage(
            getattr(completion, "usage", None),
        )

        return LLMResponse(
            text=text,
            model=request.model,
            usage=usage,
            finish_reason=getattr(
                choice,
                "finish_reason",
                None,
            ),
            metadata={
                "provider": "groq",
                "completion_id": getattr(
                    completion,
                    "id",
                    None,
                ),
            },
        )

    @staticmethod
    def _build_usage(
        usage: Any,
    ) -> LLMUsage | None:
        """Convert provider usage information into LLMUsage."""

        if usage is None:
            return None

        prompt_tokens = getattr(
            usage,
            "prompt_tokens",
            None,
        )

        completion_tokens = getattr(
            usage,
            "completion_tokens",
            None,
        )

        total_tokens = getattr(
            usage,
            "total_tokens",
            None,
        )

        if (
            prompt_tokens is None
            and completion_tokens is None
            and total_tokens is None
        ):
            return None

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _translate_exception(
        exc: Exception,
    ) -> Exception:
        """
        Translate Groq/HTTP failures into provider-independent errors.

        The SDK's concrete exception hierarchy can evolve, so translation
        uses the stable HTTP/status attributes where available and falls
        back to the provider error category.
        """

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code == 401:
            return LLMAuthenticationError(
                "Groq authentication failed.",
                provider="groq",
            )

        if status_code == 429:
            return LLMRateLimitError(
                "Groq rate limit exceeded.",
                provider="groq",
            )

        if status_code is not None and status_code >= 500:
            return LLMProviderError(
                "Groq provider returned a server error.",
                provider="groq",
                retryable=True,
            )

        if isinstance(exc, TimeoutError):
            return LLMTimeoutError(
                "Groq request timed out.",
                provider="groq",
            )

        if isinstance(exc, ValueError):
            return LLMConfigurationError(
                str(exc) or "Invalid Groq request configuration.",
                provider="groq",
            )

        return LLMProviderError(
            str(exc) or "Groq provider request failed.",
            provider="groq",
        )