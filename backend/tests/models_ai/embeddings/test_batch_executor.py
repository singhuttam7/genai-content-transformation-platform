from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingPort,
    EmbeddingRequest,
    EmbeddingVector,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchExecutor,
    EmbeddingBatchPlanner,
)


class RecordingEmbeddingProvider(EmbeddingPort):
    """Fake provider that records every batch it receives."""

    def __init__(self) -> None:
        self.model = EmbeddingModelInfo(
            provider="fake",
            model_name="fake-model",
            dimension=3,
            normalized=True,
        )
        self.requests: list[EmbeddingBatchRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        return EmbeddingVector(
            values=[0.1, 0.2, 0.3],
            model=self.model,
            input_index=0,
        )

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        self.requests.append(request)

        embeddings = [
            EmbeddingVector(
                values=[
                    float(index),
                    float(index + 1),
                    float(index + 2),
                ],
                model=self.model,
                input_index=index,
            )
            for index, _ in enumerate(request.requests)
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=self.model,
            input_count=len(request.requests),
            dimension=self.model.dimension,
        )


class FailingOnBatchProvider(EmbeddingPort):
    """Fake provider that fails on a configured batch."""

    def __init__(self, failing_batch: int) -> None:
        self.failing_batch = failing_batch
        self.call_count = 0

        self.model = EmbeddingModelInfo(
            provider="fake",
            model_name="failing-model",
            dimension=2,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise EmbeddingBatchError("Single embedding is unsupported.")

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        current_batch = self.call_count
        self.call_count += 1

        if current_batch == self.failing_batch:
            raise RuntimeError("Simulated provider failure.")

        embeddings = [
            EmbeddingVector(
                values=[0.1, 0.2],
                model=self.model,
                input_index=index,
            )
            for index, _ in enumerate(request.requests)
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=self.model,
            input_count=len(request.requests),
            dimension=2,
        )


class ExplicitBatchErrorProvider(EmbeddingPort):
    """Fake provider that already raises EmbeddingBatchError."""

    def __init__(self) -> None:
        self.model = EmbeddingModelInfo(
            provider="fake",
            model_name="error-model",
            dimension=2,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise EmbeddingBatchError("Unsupported.")

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        raise EmbeddingBatchError("Explicit batch failure.")


def make_requests(count: int) -> list[EmbeddingRequest]:
    return [
        EmbeddingRequest(
            text=f"chunk {index}",
            metadata={
                "original_index": index,
            },
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_executor_executes_all_planned_batches() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    requests = make_requests(10)

    plan = planner.plan(
        requests,
        batch_size=3,
    )

    results = await executor.execute(plan)

    assert len(results) == 4
    assert len(provider.requests) == 4


@pytest.mark.asyncio
async def test_executor_preserves_batch_sizes() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    requests = make_requests(10)

    plan = planner.plan(
        requests,
        batch_size=3,
    )

    await executor.execute(plan)

    assert [
        len(request.requests)
        for request in provider.requests
    ] == [3, 3, 3, 1]


@pytest.mark.asyncio
async def test_executor_passes_embedding_batch_requests() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(5),
        batch_size=2,
    )

    await executor.execute(plan)

    assert all(
        isinstance(request, EmbeddingBatchRequest)
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_executor_preserves_batch_order() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(7),
        batch_size=3,
    )

    results = await executor.execute(plan)

    assert len(results) == 3

    assert [result.input_count for result in results] == [
        3,
        3,
        1,
    ]


@pytest.mark.asyncio
async def test_executor_returns_batch_results() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(4),
        batch_size=2,
    )

    results = await executor.execute(plan)

    assert all(
        isinstance(result, EmbeddingBatchResult)
        for result in results
    )


@pytest.mark.asyncio
async def test_executor_preserves_provider_model_information() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(4),
        batch_size=2,
    )

    results = await executor.execute(plan)

    assert all(
        result.model.provider == "fake"
        for result in results
    )

    assert all(
        result.model.model_name == "fake-model"
        for result in results
    )


@pytest.mark.asyncio
async def test_executor_does_not_mutate_plan_batches() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    requests = make_requests(6)

    plan = planner.plan(
        requests,
        batch_size=2,
    )

    original_batches = [
        list(batch)
        for batch in plan.batches
    ]

    await executor.execute(plan)

    assert plan.batches == original_batches


@pytest.mark.asyncio
async def test_executor_does_not_mutate_original_requests() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    requests = make_requests(6)
    original = list(requests)

    plan = planner.plan(
        requests,
        batch_size=2,
    )

    await executor.execute(plan)

    assert requests == original


@pytest.mark.asyncio
async def test_executor_stops_after_provider_failure() -> None:
    provider = FailingOnBatchProvider(
        failing_batch=1,
    )
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(10),
        batch_size=3,
    )

    with pytest.raises(
        EmbeddingBatchError,
        match="Embedding batch 1 failed",
    ):
        await executor.execute(plan)

    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_executor_wraps_unexpected_provider_errors() -> None:
    provider = FailingOnBatchProvider(
        failing_batch=0,
    )
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(3),
        batch_size=2,
    )

    with pytest.raises(
        EmbeddingBatchError,
        match="Embedding batch 0 failed",
    ) as error:
        await executor.execute(plan)

    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_executor_does_not_double_wrap_batch_error() -> None:
    provider = ExplicitBatchErrorProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(2),
        batch_size=2,
    )

    with pytest.raises(
        EmbeddingBatchError,
        match="Explicit batch failure",
    ) as error:
        await executor.execute(plan)

    assert error.value.__cause__ is None


@pytest.mark.asyncio
async def test_executor_executes_batches_sequentially() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(9),
        batch_size=3,
    )

    results = await executor.execute(plan)

    assert len(results) == 3

    for index, request in enumerate(provider.requests):
        expected_start = index * 3
        expected_texts = [
            f"chunk {expected_start + offset}"
            for offset in range(len(request.requests))
        ]

        assert [
            item.text
            for item in request.requests
        ] == expected_texts


@pytest.mark.asyncio
async def test_executor_handles_single_batch() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(3),
        batch_size=10,
    )

    results = await executor.execute(plan)

    assert len(results) == 1
    assert results[0].input_count == 3
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_executor_handles_exact_batch_boundary() -> None:
    provider = RecordingEmbeddingProvider()
    planner = EmbeddingBatchPlanner()
    executor = EmbeddingBatchExecutor(provider)

    plan = planner.plan(
        make_requests(6),
        batch_size=3,
    )

    results = await executor.execute(plan)

    assert len(results) == 2
    assert [result.input_count for result in results] == [
        3,
        3,
    ]