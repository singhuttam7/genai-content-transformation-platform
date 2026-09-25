from __future__ import annotations


class LLMError(Exception):
    """Base exception for all provider-independent LLM failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)

        self.provider = provider
        self.retryable = retryable


class LLMConfigurationError(LLMError):
    """Raised when the LLM configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=False,
        )


class LLMAuthenticationError(LLMError):
    """Raised when authentication with an LLM provider fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=False,
        )


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=True,
        )


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds its timeout."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=True,
        )


class LLMProviderError(LLMError):
    """Raised for provider-side failures that are not more specific."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=retryable,
        )


class LLMResponseError(LLMError):
    """Raised when a provider returns an invalid or unusable response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=False,
        )


class LLMStructuredOutputError(LLMResponseError):
    """
    Raised when an LLM response cannot be decoded or validated
    as the requested structured output.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
        )