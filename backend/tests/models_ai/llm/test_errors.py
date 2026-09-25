from __future__ import annotations

import pytest

from app.models_ai.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


class TestLLMError:
    def test_base_error(self) -> None:
        error = LLMError(
            "LLM request failed.",
            provider="test-provider",
        )

        assert str(error) == "LLM request failed."
        assert error.provider == "test-provider"
        assert error.retryable is False

    def test_provider_is_optional(self) -> None:
        error = LLMError("LLM request failed.")

        assert error.provider is None

    def test_is_exception(self) -> None:
        error = LLMError("LLM request failed.")

        assert isinstance(error, Exception)


class TestLLMConfigurationError:
    def test_configuration_error(self) -> None:
        error = LLMConfigurationError(
            "Missing model configuration.",
            provider="test-provider",
        )

        assert str(error) == "Missing model configuration."
        assert error.provider == "test-provider"
        assert error.retryable is False
        assert isinstance(error, LLMError)


class TestLLMAuthenticationError:
    def test_authentication_error(self) -> None:
        error = LLMAuthenticationError(
            "Invalid API key.",
            provider="test-provider",
        )

        assert str(error) == "Invalid API key."
        assert error.provider == "test-provider"
        assert error.retryable is False
        assert isinstance(error, LLMError)


class TestLLMRateLimitError:
    def test_rate_limit_is_retryable(self) -> None:
        error = LLMRateLimitError(
            "Rate limit exceeded.",
            provider="test-provider",
        )

        assert str(error) == "Rate limit exceeded."
        assert error.provider == "test-provider"
        assert error.retryable is True
        assert isinstance(error, LLMError)


class TestLLMTimeoutError:
    def test_timeout_is_retryable(self) -> None:
        error = LLMTimeoutError(
            "Request timed out.",
            provider="test-provider",
        )

        assert str(error) == "Request timed out."
        assert error.provider == "test-provider"
        assert error.retryable is True
        assert isinstance(error, LLMError)


class TestLLMProviderError:
    def test_provider_error_defaults_to_non_retryable(self) -> None:
        error = LLMProviderError(
            "Provider returned an error.",
            provider="test-provider",
        )

        assert str(error) == "Provider returned an error."
        assert error.provider == "test-provider"
        assert error.retryable is False
        assert isinstance(error, LLMError)

    def test_provider_error_can_be_retryable(self) -> None:
        error = LLMProviderError(
            "Temporary provider failure.",
            provider="test-provider",
            retryable=True,
        )

        assert error.retryable is True


class TestLLMResponseError:
    def test_response_error(self) -> None:
        error = LLMResponseError(
            "Provider returned invalid response.",
            provider="test-provider",
        )

        assert str(error) == (
            "Provider returned invalid response."
        )
        assert error.provider == "test-provider"
        assert error.retryable is False
        assert isinstance(error, LLMError)


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        "error_type",
        [
            LLMConfigurationError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
            LLMResponseError,
        ],
    )
    def test_all_errors_inherit_from_llm_error(
        self,
        error_type: type[LLMError],
    ) -> None:
        error = error_type("Test error.")

        assert isinstance(error, LLMError)
        assert isinstance(error, Exception)