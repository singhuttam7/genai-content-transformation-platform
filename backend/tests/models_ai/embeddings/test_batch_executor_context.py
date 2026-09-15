from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelConfig,
    EmbeddingModelInfo,
    EmbeddingOperation,
    EmbeddingRequest,
    EmbeddingVector,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchExecutor,
    EmbeddingBatchPlanner,
)
from app.models_ai.embeddings.port import EmbeddingPort


class ContextFailingProvider(EmbeddingPort):
    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[EmbeddingBatchRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise RuntimeError("single embedding failure")

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        call_index = len(self.calls)
        self.calls.append(request)

        if call_index == self.fail_on_call:
            raise RuntimeError("simulated provider failure")

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


def make_requests(
    count: int,
) -> list[EmbeddingRequest]:
    return [
        EmbeddingRequest(
            text=f"chunk {index}"
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
async def test_executor_attaches_batch_context() -> None:
    provider = ContextFailingProvider(
        fail_on_call=1,
    )

    model_config = EmbeddingModelConfig(
        provider="local_sentence_transformer",
        model_name="all-MiniLM-L6-v2",
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=model_config,
    )

    plan = make_plan(
        count=10,
        batch_size=3,
    )

    with pytest.raises(
        EmbeddingBatchError,
    ) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.batch_index == 1
    assert error.context is not None
    assert error.context.operation == EmbeddingOperation.EMBED_BATCH
    assert error.context.provider == "local_sentence_transformer"
    assert error.context.model == "all-MiniLM-L6-v2"
    assert error.context.batch_index == 1
    assert error.context.input_count == 3


@pytest.mark.asyncio
async def test_executor_context_matches_batch_index() -> None:
    provider = ContextFailingProvider(
        fail_on_call=2,
    )

    model_config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=model_config,
    )

    plan = make_plan(
        count=10,
        batch_size=2,
    )

    with pytest.raises(
        EmbeddingBatchError,
    ) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.batch_index == 2
    assert error.context is not None
    assert error.context.batch_index == 2


@pytest.mark.asyncio
async def test_executor_context_input_count_matches_failed_batch() -> None:
    provider = ContextFailingProvider(
        fail_on_call=2,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
        ),
    )

    plan = make_plan(
        count=7,
        batch_size=3,
    )

    with pytest.raises(
        EmbeddingBatchError,
    ) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None
    assert error.context.batch_index == 2
    assert error.context.input_count == 1


@pytest.mark.asyncio
async def test_executor_leaves_provider_and_model_unknown_without_config() -> None:
    provider = ContextFailingProvider(
        fail_on_call=0,
    )

    executor = EmbeddingBatchExecutor(provider)

    plan = make_plan(
        count=4,
        batch_size=2,
    )

    with pytest.raises(
        EmbeddingBatchError,
    ) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None
    assert error.context.provider is None
    assert error.context.model is None


@pytest.mark.asyncio
async def test_executor_preserves_model_options_in_context() -> None:
    provider = ContextFailingProvider(
        fail_on_call=0,
    )

    model_config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        options={
            "device": "cpu",
            "normalize_embeddings": True,
        },
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=model_config,
    )

    plan = make_plan(
        count=2,
        batch_size=2,
    )

    with pytest.raises(
        EmbeddingBatchError,
    ) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None
    assert error.context.metadata["model_options"] == {
        "device": "cpu",
        "normalize_embeddings": True,
    }


@pytest.mark.asyncio
async def test_executor_does_not_mutate_model_options() -> None:
    provider = ContextFailingProvider(
        fail_on_call=0,
    )

    options = {
        "device": "cpu",
        "batch_size": 16,
    }

    model_config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        options=options,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=model_config,
    )

    plan = make_plan(
        count=2,
        batch_size=2,
    )

    with pytest.raises(EmbeddingBatchError):
        await executor.execute(plan)

    assert model_config.options == options


@pytest.mark.asyncio
async def test_executor_stops_after_contextualized_failure() -> None:
    provider = ContextFailingProvider(
        fail_on_call=1,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
        ),
    )

    plan = make_plan(
        count=20,
        batch_size=2,
    )

    with pytest.raises(EmbeddingBatchError):
        await executor.execute(plan)

    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_original_exception_is_preserved() -> None:
    provider = ContextFailingProvider(
        fail_on_call=0,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
        ),
    )

    plan = make_plan(
        count=2,
        batch_size=2,
    )

    with pytest.raises(
        EmbeddingBatchError,
    ) as exc_info:
        await executor.execute(plan)

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    assert str(exc_info.value.__cause__) == (
        "simulated provider failure"
    )