from __future__ import annotations
import asyncio

from unittest.mock import AsyncMock, Mock

import pytest

from app.models_ai.llm.errors import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
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


MODEL = LLMModelInfo(
    provider="test-provider",
    model_name="test-model",
)


def build_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Hello.",
            ),
        ],
        model=MODEL,
    )


def build_response() -> LLMResponse:
    return LLMResponse(
        text="Hello back.",
        model=MODEL,
    )


def build_port() -> Mock:
    return Mock(
        spec=LLMPort,
    )


def build_service(
    *,
    port: Mock,
    policy: LLMRetryPolicy | None = None,
    sleep: AsyncMock | None = None,
    timeout_seconds: float | None = None,
) -> ResilientLLMPort:
    return ResilientLLMPort(
        port,
        retry_policy=policy,
        sleep=sleep,
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.asyncio
async def test_first_attempt_success_does_not_retry() -> None:
    port = build_port()

    port.generate = AsyncMock(
        return_value=build_response(),
    )

    service = build_service(
        port=port,
    )

    response = await service.generate(
        build_request(),
    )

    assert response.text == "Hello back."
    port.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_retryable_failure_then_success() -> None:
    port = build_port()

    port.generate = AsyncMock(
        side_effect=[
            LLMRateLimitError(
                "Rate limited.",
                provider="test-provider",
            ),
            build_response(),
        ],
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.5,
        backoff_multiplier=2.0,
        max_backoff_seconds=8.0,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
    )

    response = await service.generate(
        build_request(),
    )

    assert response.text == "Hello back."
    assert port.generate.await_count == 2

    sleep.assert_awaited_once_with(
        0.5,
    )


@pytest.mark.asyncio
async def test_retryable_failure_uses_increasing_backoff() -> None:
    port = build_port()

    error = LLMProviderError(
        "Temporary provider failure.",
        provider="test-provider",
        retryable=True,
    )

    port.generate = AsyncMock(
        side_effect=[
            error,
            error,
            build_response(),
        ],
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.5,
        backoff_multiplier=2.0,
        max_backoff_seconds=8.0,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
    )

    response = await service.generate(
        build_request(),
    )

    assert response.text == "Hello back."
    assert port.generate.await_count == 3

    assert [
        call.args[0]
        for call in sleep.await_args_list
    ] == [
        0.5,
        1.0,
    ]


@pytest.mark.asyncio
async def test_retryable_failure_is_exhausted() -> None:
    port = build_port()

    error = LLMProviderError(
        "Provider unavailable.",
        provider="test-provider",
        retryable=True,
    )

    port.generate = AsyncMock(
        side_effect=error,
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.1,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is error
    assert port.generate.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried() -> None:
    port = build_port()

    error = LLMAuthenticationError(
        "Authentication failed.",
        provider="test-provider",
    )

    port.generate = AsyncMock(
        side_effect=error,
    )

    sleep = AsyncMock()

    service = build_service(
        port=port,
        sleep=sleep,
    )

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is error
    port.generate.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_error_from_provider_is_retryable() -> None:
    port = build_port()

    port.generate = AsyncMock(
        side_effect=[
            LLMTimeoutError(
                "Request timed out.",
                provider="test-provider",
            ),
            build_response(),
        ],
    )

    sleep = AsyncMock()

    service = build_service(
        port=port,
        sleep=sleep,
    )

    response = await service.generate(
        build_request(),
    )

    assert response.text == "Hello back."
    assert port.generate.await_count == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_retryable_provider_error_without_retry_budget_is_not_retried() -> None:
    port = build_port()

    error = LLMProviderError(
        "Temporary failure.",
        provider="test-provider",
        retryable=True,
    )

    port.generate = AsyncMock(
        side_effect=error,
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=1,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is error
    port.generate.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_is_passed_without_mutation() -> None:
    port = build_port()

    port.generate = AsyncMock(
        return_value=build_response(),
    )

    service = build_service(
        port=port,
    )

    request = build_request()

    original_messages = list(
        request.messages,
    )
    original_metadata = dict(
        request.metadata,
    )

    await service.generate(request)

    assert request.messages == original_messages
    assert request.metadata == original_metadata

    port.generate.assert_awaited_once_with(
        request,
    )


@pytest.mark.asyncio
async def test_platform_timeout_is_translated_to_llm_timeout_error() -> None:
    port = build_port()

    async def slow_generate(
        request: LLMRequest,
    ) -> LLMResponse:
        await __import__("asyncio").sleep(10)
        return build_response()

    port.generate = AsyncMock(
        side_effect=slow_generate,
    )

    service = build_service(
        port=port,
        timeout_seconds=0.01,
    )

    with pytest.raises(LLMTimeoutError) as exc_info:
        await service.generate(
            build_request(),
        )

    error = exc_info.value

    assert error.provider == "test-provider"
    assert error.retryable is True
    assert port.generate.await_count == 3


@pytest.mark.asyncio
async def test_platform_timeout_can_be_retried() -> None:
    port = build_port()

    attempts = 0

    async def generate_with_one_timeout(
        request: LLMRequest,
    ) -> LLMResponse:
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            await __import__("asyncio").sleep(10)

        return build_response()

    port.generate = AsyncMock(
        side_effect=generate_with_one_timeout,
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=2,
        initial_backoff_seconds=0.5,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
        timeout_seconds=0.01,
    )

    response = await service.generate(
        build_request(),
    )

    assert response.text == "Hello back."
    assert port.generate.await_count == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_platform_timeout_is_exhausted() -> None:
    port = build_port()

    async def slow_generate(
        request: LLMRequest,
    ) -> LLMResponse:
        await __import__("asyncio").sleep(10)
        return build_response()

    port.generate = AsyncMock(
        side_effect=slow_generate,
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.01,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
        timeout_seconds=0.01,
    )

    with pytest.raises(LLMTimeoutError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value.provider == "test-provider"
    assert exc_info.value.retryable is True
    assert port.generate.await_count == 3
    assert sleep.await_count == 2


def test_port_must_implement_llm_port() -> None:
    with pytest.raises(TypeError) as exc_info:
        ResilientLLMPort(
            Mock(),
        )

    assert str(exc_info.value) == (
        "port must be an LLMPort."
    )


def test_default_policy_is_created() -> None:
    port = build_port()

    service = ResilientLLMPort(
        port,
    )

    assert isinstance(
        service.retry_policy,
        LLMRetryPolicy,
    )


def test_custom_policy_is_preserved() -> None:
    port = build_port()

    policy = LLMRetryPolicy(
        max_attempts=5,
        initial_backoff_seconds=1.0,
        backoff_multiplier=1.5,
        max_backoff_seconds=10.0,
    )

    service = ResilientLLMPort(
        port,
        retry_policy=policy,
    )

    assert service.retry_policy is policy
    assert service.port is port


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
        -0.01,
    ],
)
def test_invalid_timeout_is_rejected(
    timeout_seconds: float,
) -> None:
    port = build_port()

    with pytest.raises(ValueError) as exc_info:
        ResilientLLMPort(
            port,
            timeout_seconds=timeout_seconds,
        )

    assert str(exc_info.value) == (
        "timeout_seconds must be greater than 0."
    )


def test_timeout_defaults_to_none() -> None:
    port = build_port()

    service = ResilientLLMPort(
        port,
    )

    assert service.timeout_seconds is None


def test_timeout_configuration_is_preserved() -> None:
    port = build_port()

    service = ResilientLLMPort(
        port,
        timeout_seconds=30.0,
    )

    assert service.timeout_seconds == 30.0

@pytest.mark.asyncio
async def test_unexpected_exception_is_not_retried() -> None:
    port = build_port()

    error = RuntimeError(
        "Unexpected provider integration failure.",
    )

    port.generate = AsyncMock(
        side_effect=error,
    )

    sleep = AsyncMock()

    service = build_service(
        port=port,
        sleep=sleep,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is error
    port.generate.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_value_error_is_not_retried() -> None:
    port = build_port()

    error = ValueError(
        "Invalid provider response.",
    )

    port.generate = AsyncMock(
        side_effect=error,
    )

    sleep = AsyncMock()

    service = build_service(
        port=port,
        sleep=sleep,
    )

    with pytest.raises(ValueError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is error
    port.generate.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_error_is_not_retried() -> None:
    port = build_port()

    port.generate = AsyncMock(
        side_effect=asyncio.CancelledError(),
    )

    sleep = AsyncMock()

    service = build_service(
        port=port,
        sleep=sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.generate(
            build_request(),
        )

    port.generate.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_retryable_llm_error_preserves_original_instance() -> None:
    port = build_port()

    error = LLMAuthenticationError(
        "Authentication failed.",
        provider="test-provider",
    )

    port.generate = AsyncMock(
        side_effect=error,
    )

    service = build_service(
        port=port,
    )

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is error
    assert exc_info.value.provider == "test-provider"
    assert exc_info.value.retryable is False
    port.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_retryable_error_preserves_final_original_instance() -> None:
    port = build_port()

    first_error = LLMProviderError(
        "Temporary failure.",
        provider="test-provider",
        retryable=True,
    )

    final_error = LLMProviderError(
        "Final provider failure.",
        provider="test-provider",
        retryable=True,
    )

    port.generate = AsyncMock(
        side_effect=[
            first_error,
            final_error,
        ],
    )

    sleep = AsyncMock()

    policy = LLMRetryPolicy(
        max_attempts=2,
        initial_backoff_seconds=0.1,
    )

    service = build_service(
        port=port,
        policy=policy,
        sleep=sleep,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await service.generate(
            build_request(),
        )

    assert exc_info.value is final_error
    assert port.generate.await_count == 2
    sleep.assert_awaited_once_with(0.1)