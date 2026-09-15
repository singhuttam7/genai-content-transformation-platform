from __future__ import annotations

from typing import Any

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchExecutor,
    EmbeddingBatchPlanner,
)
from app.models_ai.embeddings.port import EmbeddingPort


class SuccessfulProvider(EmbeddingPort):
    """Provider that succeeds for every batch."""

    def __init__(self) -> None:
        self.calls: list[EmbeddingBatchRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        return EmbeddingVector(
            values=[1.0, 2.0],
            model=EmbeddingModelInfo(
                provider="fake",
                model_name="fake-model",
                dimension=2,
            ),
            input_index=0,
        )

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        self.calls.append(request)

        model = EmbeddingModelInfo(
            provider="fake",
            model_name="fake-model",
            dimension=2,
        )

        embeddings = [
            EmbeddingVector(
                values=[float(index), float(index + 1)],
                model=model,
                input_index=index,
            )
            for index in range(len(request.requests))
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=model,
            input_count=len(request.requests),
            dimension=2,
        )


class FailingProvider(EmbeddingPort):
    """Provider that fails on a configured batch call."""

    def __init__(
        self,
        *,
        fail_on_call: int,
        exception: Exception | None = None,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.exception = exception or RuntimeError(
            "simulated provider failure"
        )
        self.calls: list[EmbeddingBatchRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise self.exception

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        call_index = len(self.calls)
        self.calls.append(request)

        if call_index == self.fail_on_call:
            raise self.exception

        model = EmbeddingModelInfo(
            provider="fake",
            model_name="fake-model",
            dimension=2,
        )

        embeddings = [
            EmbeddingVector(
                values=[1.0, 2.0],
                model=model,
                input_index=index,
            )
            for index in range(len(request.requests))
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=model,
            input_count=len(request.requests),
            dimension=2,
        )


class BatchErrorProvider(EmbeddingPort):
    """Provider that raises an existing EmbeddingBatchError."""

    def __init__(self) -> None:
        self.error = EmbeddingBatchError(
            "provider supplied batch failure",
            batch_index=99,
        )
        self.calls: list[EmbeddingBatchRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise self.error

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        self.calls.append(request)
        raise self.error


def make_requests(
    count: int,
) -> list[EmbeddingRequest]:
    return [
        EmbeddingRequest(
            text=f"chunk {index}",
            metadata={"index": index},
        )
        for index in range(count)
    ]


def make_plan(
    count: int,
    batch_size: int,
):
    planner = EmbeddingBatchPlanner()

    return planner.plan(
        make_requests(count),
        batch_size=batch_size,
    )


@pytest.mark.asyncio
async def test_all_successful_batches_return_all_results() -> None:
    provider = SuccessfulProvider()

    plan = make_plan(
        count=10,
        batch_size=3,
    )

    executor = EmbeddingBatchExecutor(provider)

    results = await executor.execute(plan)

    assert len(results) == plan.batch_count
    assert len(provider.calls) == plan.batch_count


@pytest.mark.asyncio
async def test_failure_stops_execution_at_first_failed_batch() -> None:
    provider = FailingProvider(
        fail_on_call=1,
    )

    plan = make_plan(
        count=10,
        batch_size=3,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(
        EmbeddingBatchError,
        match="Embedding batch 1 failed",
    ) as exc_info:
        await executor.execute(plan)

    assert exc_info.value.batch_index == 1
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_batches_after_failure_are_not_executed() -> None:
    provider = FailingProvider(
        fail_on_call=1,
    )

    plan = make_plan(
        count=20,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(EmbeddingBatchError):
        await executor.execute(plan)

    assert len(provider.calls) == 2
    assert len(provider.calls) < plan.batch_count


@pytest.mark.asyncio
async def test_failure_does_not_return_partial_results() -> None:
    provider = FailingProvider(
        fail_on_call=1,
    )

    plan = make_plan(
        count=10,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(
        EmbeddingBatchError,
        match="Embedding batch 1 failed",
    ):
        await executor.execute(plan)

    # Batch 0 succeeded, batch 1 failed, and execution stopped.
    # Therefore no partial result list escaped the executor.
    assert len(provider.calls) == 2

@pytest.mark.asyncio
async def test_unexpected_exception_is_wrapped() -> None:
    original_error = RuntimeError(
        "provider connection failed"
    )

    provider = FailingProvider(
        fail_on_call=0,
        exception=original_error,
    )

    plan = make_plan(
        count=3,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(
        EmbeddingBatchError,
        match="Embedding batch 0 failed",
    ) as exc_info:
        await executor.execute(plan)

    assert exc_info.value.batch_index == 0
    assert exc_info.value.__cause__ is original_error


@pytest.mark.asyncio
async def test_existing_batch_error_is_not_double_wrapped() -> None:
    provider = BatchErrorProvider()

    plan = make_plan(
        count=2,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(
        EmbeddingBatchError
    ) as exc_info:
        await executor.execute(plan)

    assert exc_info.value is provider.error
    assert exc_info.value.batch_index == 99


@pytest.mark.asyncio
async def test_failure_on_first_batch_executes_only_first_batch() -> None:
    provider = FailingProvider(
        fail_on_call=0,
    )

    plan = make_plan(
        count=100,
        batch_size=10,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(EmbeddingBatchError):
        await executor.execute(plan)

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_failure_on_last_batch_executes_all_batches_until_failure() -> None:
    plan = make_plan(
        count=10,
        batch_size=3,
    )

    last_batch_index = plan.batch_count - 1

    provider = FailingProvider(
        fail_on_call=last_batch_index,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    assert exc_info.value.batch_index == last_batch_index
    assert len(provider.calls) == plan.batch_count


@pytest.mark.asyncio
async def test_failure_index_is_zero_based() -> None:
    provider = FailingProvider(
        fail_on_call=2,
    )

    plan = make_plan(
        count=10,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    assert exc_info.value.batch_index == 2


@pytest.mark.asyncio
async def test_successful_batches_are_not_exposed_on_failure() -> None:
    provider = FailingProvider(
        fail_on_call=2,
    )

    plan = make_plan(
        count=12,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(
        EmbeddingBatchError,
        match="Embedding batch 2 failed",
    ) as exc_info:
        await executor.execute(plan)

    assert exc_info.value.batch_index == 2

    # Batches 0 and 1 succeeded, batch 2 failed.
    # The executor does not return the successful intermediate results.
    assert len(provider.calls) == 3

@pytest.mark.asyncio
async def test_executor_does_not_retry_failed_batch() -> None:
    provider = FailingProvider(
        fail_on_call=1,
    )

    plan = make_plan(
        count=10,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(EmbeddingBatchError):
        await executor.execute(plan)

    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_failure_preserves_provider_exception_chain() -> None:
    original_error = ValueError(
        "invalid provider response"
    )

    provider = FailingProvider(
        fail_on_call=0,
        exception=original_error,
    )

    plan = make_plan(
        count=2,
        batch_size=2,
    )

    executor = EmbeddingBatchExecutor(provider)

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    assert isinstance(
        exc_info.value.__cause__,
        ValueError,
    )

    assert str(exc_info.value.__cause__) == (
        "invalid provider response"
    )


@pytest.mark.asyncio
async def test_successful_execution_is_deterministic() -> None:
    plan = make_plan(
        count=7,
        batch_size=3,
    )

    provider = SuccessfulProvider()
    executor = EmbeddingBatchExecutor(provider)

    first = await executor.execute(plan)

    provider.calls.clear()

    second = await executor.execute(plan)

    assert first == second


def test_batch_error_rejects_negative_batch_index() -> None:
    with pytest.raises(
        ValueError,
        match="batch_index cannot be negative",
    ):
        EmbeddingBatchError(
            "invalid batch",
            batch_index=-1,
        )


def test_batch_error_allows_unknown_batch_index() -> None:
    error = EmbeddingBatchError(
        "unknown batch",
    )

    assert error.batch_index is None


def test_batch_error_stores_batch_index() -> None:
    error = EmbeddingBatchError(
        "batch failed",
        batch_index=5,
    )

    assert error.batch_index == 5


def test_batch_error_is_an_embedding_error() -> None:
    from app.models_ai.embeddings import EmbeddingError

    error = EmbeddingBatchError("batch failed")

    assert isinstance(error, EmbeddingError)