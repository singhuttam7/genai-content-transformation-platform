from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.models_ai.llm.errors import LLMError, LLMTimeoutError
from app.models_ai.llm.port import LLMPort
from app.models_ai.llm.resilience import LLMRetryPolicy
from app.models_ai.llm.schemas import LLMRequest, LLMResponse


SleepFunction = Callable[[float], Awaitable[None]]


class ResilientLLMPort(LLMPort):
    """
    Provider-independent retry and timeout wrapper around an LLMPort.

    Retry decisions are based exclusively on the provider-independent
    LLMError.retryable flag and LLMRetryPolicy.

    timeout_seconds applies independently to each provider attempt.
    """

    def __init__(
        self,
        port: LLMPort,
        *,
        retry_policy: LLMRetryPolicy | None = None,
        sleep: SleepFunction | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if not isinstance(port, LLMPort):
            raise TypeError(
                "port must be an LLMPort."
            )

        if (
            timeout_seconds is not None
            and timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        self._port = port
        self._retry_policy = (
            retry_policy
            if retry_policy is not None
            else LLMRetryPolicy()
        )
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds

    @property
    def port(self) -> LLMPort:
        """Return the wrapped provider-independent port."""

        return self._port

    @property
    def retry_policy(self) -> LLMRetryPolicy:
        """Return the configured retry policy."""

        return self._retry_policy

    @property
    def timeout_seconds(self) -> float | None:
        """Return the configured per-attempt timeout."""

        return self._timeout_seconds

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate an LLM response with bounded retry and timeout behavior.

        Each provider attempt receives the configured timeout.

        Only LLMError instances marked retryable are retried.
        Non-retryable errors are propagated immediately.
        """

        attempt_number = 1

        while True:
            try:
                response = await self._generate_with_timeout(
                    request,
                )
                return response

            except LLMError as exc:
                if not self._retry_policy.should_retry(
                    attempt_number=attempt_number,
                    retryable=exc.retryable,
                ):
                    raise

                delay = self._retry_policy.backoff_seconds(
                    attempt_number=attempt_number,
                )

                if delay > 0:
                    await self._wait(delay)

                attempt_number += 1

    async def _generate_with_timeout(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Execute one provider attempt with the platform timeout.

        Provider-independent LLM errors are preserved unchanged.
        asyncio timeout is translated into LLMTimeoutError.
        """

        if self._timeout_seconds is None:
            return await self._port.generate(request)

        try:
            return await asyncio.wait_for(
                self._port.generate(request),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(
                "LLM request timed out.",
                provider=request.model.provider,
            ) from exc

    async def _wait(
        self,
        delay_seconds: float,
    ) -> None:
        """
        Wait before the next retry.

        A custom sleeper can be injected for deterministic tests.
        """

        if self._sleep is not None:
            await self._sleep(delay_seconds)
            return

        await asyncio.sleep(delay_seconds)