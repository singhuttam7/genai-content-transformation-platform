from __future__ import annotations

import json

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingErrorCategory,
    EmbeddingErrorContext,
    EmbeddingModelConfig,
    EmbeddingOperation,
    EmbeddingPort,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchExecutor,
    EmbeddingBatchPlanner,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)


class FailingEmbeddingProvider(EmbeddingPort):
    """Fake provider that fails on a configured batch."""

    def __init__(self, failing_batch_index: int) -> None:
        self.failing_batch_index = failing_batch_index
        self.call_count = 0

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise RuntimeError("Single embedding operation failed.")

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        current_batch_index = self.call_count
        self.call_count += 1

        if current_batch_index == self.failing_batch_index:
            raise RuntimeError(
                f"Provider failed on batch {current_batch_index}."
            )

        model = EmbeddingModelInfo(
            provider="fake",
            model_name="fake-model",
            dimension=3,
        )

        embeddings = [
            EmbeddingVector(
                values=[1.0, 2.0, 3.0],
                model=model,
                input_index=index,
            )
            for index, _ in enumerate(request.requests)
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=model,
            input_count=len(request.requests),
            dimension=3,
        )


def create_plan() -> object:
    requests = [
        EmbeddingRequest(text=f"text-{index}")
        for index in range(5)
    ]

    planner = EmbeddingBatchPlanner()

    return planner.plan(
        requests,
        batch_size=2,
    )


@pytest.mark.asyncio
async def test_executor_attaches_complete_context_to_unexpected_error() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=1,
    )

    model_config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        expected_dimension=3,
        batch_size=2,
        options={
            "normalize": True,
        },
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=model_config,
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.batch_index == 1
    assert error.context is not None

    assert error.context.operation == EmbeddingOperation.EMBED_BATCH
    assert error.context.provider == "fake"
    assert error.context.model == "fake-model"
    assert error.context.batch_index == 1
    assert error.context.input_count == 2
    assert error.context.metadata == {
        "model_options": {
            "normalize": True,
        }
    }


@pytest.mark.asyncio
async def test_error_category_remains_batch() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=0,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.category == EmbeddingErrorCategory.BATCH


@pytest.mark.asyncio
async def test_original_provider_exception_is_preserved_as_cause() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=0,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.__cause__ is not None
    assert isinstance(error.__cause__, RuntimeError)
    assert str(error.__cause__) == (
        "Provider failed on batch 0."
    )


@pytest.mark.asyncio
async def test_error_context_can_be_serialized_after_execution_failure() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=1,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None

    payload = error.context.model_dump(mode="json")

    assert payload["operation"] == "embed_batch"
    assert payload["provider"] == "fake"
    assert payload["model"] == "fake-model"
    assert payload["batch_index"] == 1
    assert payload["input_count"] == 2


@pytest.mark.asyncio
async def test_serialized_context_can_be_reconstructed() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=1,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None

    serialized = json.dumps(
        error.context.model_dump(mode="json")
    )

    restored = EmbeddingErrorContext.model_validate(
        json.loads(serialized)
    )

    assert restored == error.context


@pytest.mark.asyncio
async def test_failure_stops_execution_at_first_failed_batch() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=1,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError):
        await executor.execute(plan)

    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_failure_context_identifies_correct_batch() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=2,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.batch_index == 2
    assert error.context is not None
    assert error.context.batch_index == 2
    assert error.context.input_count == 1


@pytest.mark.asyncio
async def test_context_without_model_config_remains_serializable() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=0,
    )

    executor = EmbeddingBatchExecutor(provider)

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None

    payload = error.context.model_dump(mode="json")

    assert payload["operation"] == "embed_batch"
    assert payload["provider"] is None
    assert payload["model"] is None
    assert payload["batch_index"] == 0
    assert payload["input_count"] == 2


@pytest.mark.asyncio
async def test_context_metadata_does_not_expose_provider_object() -> None:
    provider = FailingEmbeddingProvider(
        failing_batch_index=0,
    )

    executor = EmbeddingBatchExecutor(
        provider,
        model_config=EmbeddingModelConfig(
            provider="fake",
            model_name="fake-model",
            batch_size=2,
        ),
    )

    plan = create_plan()

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await executor.execute(plan)

    error = exc_info.value

    assert error.context is not None

    serialized = error.context.model_dump(mode="json")

    assert serialized["metadata"] == {}