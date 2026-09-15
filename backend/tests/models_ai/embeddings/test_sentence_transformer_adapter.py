from __future__ import annotations

from typing import Any

import pytest

from app.models_ai.embeddings.adapters.sentence_transformer import (
    SentenceTransformerAdapter,
)
from app.models_ai.embeddings.exceptions import (
    EmbeddingDimensionError,
    EmbeddingErrorCategory,
    EmbeddingProviderError,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingModelConfig,
    EmbeddingRequest,
)


class FakeSentenceTransformer:
    """Deterministic fake Sentence-Transformer runtime for unit tests."""

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int = 3,
        encode_error: Exception | None = None,
        output: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.encode_error = encode_error
        self.output = output

        self.encode_calls: list[dict[str, Any]] = []

    def encode(
        self,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> Any:
        self.encode_calls.append(
            {
                "inputs": inputs,
                "kwargs": kwargs,
            }
        )

        if self.encode_error is not None:
            raise self.encode_error

        if self.output is not None:
            return self.output

        if isinstance(inputs, str):
            return [0.1, 0.2, 0.3][: self.dimension]

        return [
            [float(index + 1), 0.2, 0.3][: self.dimension]
            for index, _ in enumerate(inputs)
        ]


class FakeModelFactory:
    """Tracks model construction for lifecycle tests."""

    def __init__(self, model: FakeSentenceTransformer) -> None:
        self.model = model
        self.calls: list[str] = []

    def __call__(self, model_name: str) -> FakeSentenceTransformer:
        self.calls.append(model_name)
        return self.model


def create_config(
    *,
    expected_dimension: int | None = 3,
    normalized: bool = False,
    batch_size: int | None = None,
    options: dict[str, object] | None = None,
) -> EmbeddingModelConfig:
    config_kwargs: dict[str, object] = {
        "provider": "sentence-transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "expected_dimension": expected_dimension,
        "normalized": normalized,
        "options": options or {},
    }

    if batch_size is not None:
        config_kwargs["batch_size"] = batch_size

    return EmbeddingModelConfig(**config_kwargs)

def create_adapter(
    model: FakeSentenceTransformer,
    *,
    expected_dimension: int | None = 3,
    normalized: bool = False,
    batch_size: int | None = None,
    options: dict[str, object] | None = None,
) -> tuple[SentenceTransformerAdapter, FakeModelFactory]:
    factory = FakeModelFactory(model)

    adapter = SentenceTransformerAdapter(
        create_config(
            expected_dimension=expected_dimension,
            normalized=normalized,
            batch_size=batch_size,
            options=options,
        ),
        model_factory=factory,
    )

    return adapter, factory


def test_adapter_does_not_load_model_during_construction() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, factory = create_adapter(model)

    assert adapter.is_loaded is False
    assert adapter.model is None
    assert factory.calls == []


@pytest.mark.asyncio
async def test_single_embedding_loads_model_lazily() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, factory = create_adapter(model)

    result = await adapter.embed(
        EmbeddingRequest(text="hello world")
    )

    assert adapter.is_loaded is True
    assert adapter.model is model
    assert factory.calls == [
        "sentence-transformers/all-MiniLM-L6-v2"
    ]

    assert result.values == [0.1, 0.2, 0.3]
    assert result.model.provider == "sentence-transformers"
    assert result.model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.model.dimension == 3
    assert result.model.normalized is False
    assert result.input_index == 0


@pytest.mark.asyncio
async def test_loaded_model_is_reused_for_multiple_requests() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, factory = create_adapter(model)

    await adapter.embed(EmbeddingRequest(text="first"))
    await adapter.embed(EmbeddingRequest(text="second"))

    assert factory.calls == [
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    assert model.encode_calls.__len__() == 2


@pytest.mark.asyncio
async def test_single_embedding_passes_text_to_model() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(model)

    await adapter.embed(
        EmbeddingRequest(text="hello world")
    )

    assert model.encode_calls[0]["inputs"] == "hello world"


@pytest.mark.asyncio
async def test_normalization_option_is_passed_to_model() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        normalized=True,
    )

    await adapter.embed(
        EmbeddingRequest(text="hello")
    )

    assert model.encode_calls[0]["kwargs"][
        "normalize_embeddings"
    ] is True


@pytest.mark.asyncio
async def test_batch_size_option_is_passed_to_model() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        batch_size=16,
    )

    await adapter.embed(
        EmbeddingRequest(text="hello")
    )

    assert model.encode_calls[0]["kwargs"]["batch_size"] == 16


@pytest.mark.asyncio
async def test_custom_options_are_passed_to_model() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        options={
            "show_progress_bar": False,
        },
    )

    await adapter.embed(
        EmbeddingRequest(text="hello")
    )

    assert model.encode_calls[0]["kwargs"][
        "show_progress_bar"
    ] is False


@pytest.mark.asyncio
async def test_explicit_option_overrides_default_normalization() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        normalized=True,
        options={
            "normalize_embeddings": False,
        },
    )

    await adapter.embed(
        EmbeddingRequest(text="hello")
    )

    assert model.encode_calls[0]["kwargs"][
        "normalize_embeddings"
    ] is False


@pytest.mark.asyncio
async def test_explicit_batch_size_option_overrides_configured_batch_size() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        batch_size=32,
        options={
            "batch_size": 8,
        },
    )

    await adapter.embed(
        EmbeddingRequest(text="hello")
    )

    assert model.encode_calls[0]["kwargs"]["batch_size"] == 8


@pytest.mark.asyncio
async def test_batch_embedding_preserves_input_order() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(model)

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="first"),
            EmbeddingRequest(text="second"),
            EmbeddingRequest(text="third"),
        ]
    )

    result = await adapter.embed_batch(request)

    assert len(result.embeddings) == 3
    assert [item.input_index for item in result.embeddings] == [
        0,
        1,
        2,
    ]

    assert result.embeddings[0].values == [1.0, 0.2, 0.3]
    assert result.embeddings[1].values == [2.0, 0.2, 0.3]
    assert result.embeddings[2].values == [3.0, 0.2, 0.3]


@pytest.mark.asyncio
async def test_batch_embedding_passes_all_texts_to_model() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(model)

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="alpha"),
            EmbeddingRequest(text="beta"),
            EmbeddingRequest(text="gamma"),
        ]
    )

    await adapter.embed_batch(request)

    assert model.encode_calls[0]["inputs"] == [
        "alpha",
        "beta",
        "gamma",
    ]


@pytest.mark.asyncio
async def test_duplicate_inputs_remain_distinct() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(model)

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="same"),
            EmbeddingRequest(text="same"),
            EmbeddingRequest(text="different"),
        ]
    )

    result = await adapter.embed_batch(request)

    assert len(result.embeddings) == 3
    assert [item.input_index for item in result.embeddings] == [
        0,
        1,
        2,
    ]


@pytest.mark.asyncio
async def test_batch_result_contains_correct_counts_and_dimension() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(model)

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="one"),
            EmbeddingRequest(text="two"),
        ]
    )

    result = await adapter.embed_batch(request)

    assert result.input_count == 2
    assert len(result.embeddings) == 2
    assert result.dimension == 3
    assert result.model.provider == "sentence-transformers"
    assert result.model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.model.dimension == 3
    assert result.model.normalized is False


@pytest.mark.asyncio
async def test_dimension_mismatch_is_rejected_for_single_embedding() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        expected_dimension=4,
    )

    with pytest.raises(EmbeddingDimensionError):
        await adapter.embed(
            EmbeddingRequest(text="hello")
        )


@pytest.mark.asyncio
async def test_dimension_mismatch_is_rejected_for_batch_embedding() -> None:
    model = FakeSentenceTransformer("fake-model")
    adapter, _ = create_adapter(
        model,
        expected_dimension=4,
    )

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="hello"),
            EmbeddingRequest(text="world"),
        ]
    )

    with pytest.raises(EmbeddingDimensionError):
        await adapter.embed_batch(request)


@pytest.mark.asyncio
async def test_model_output_is_converted_from_numpy_like_object() -> None:
    class FakeArray:
        def tolist(self) -> list[float]:
            return [0.1, 0.2, 0.3]

    model = FakeSentenceTransformer(
        "fake-model",
        output=FakeArray(),
    )

    adapter, _ = create_adapter(model)

    result = await adapter.embed(
        EmbeddingRequest(text="hello")
    )

    assert result.values == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_non_sequence_single_output_is_rejected() -> None:
    model = FakeSentenceTransformer(
        "fake-model",
        output=42,
    )

    adapter, _ = create_adapter(model)

    with pytest.raises(EmbeddingDimensionError):
        await adapter.embed(
            EmbeddingRequest(text="hello")
        )


@pytest.mark.asyncio
async def test_empty_single_vector_is_rejected() -> None:
    model = FakeSentenceTransformer(
        "fake-model",
        output=[],
    )

    adapter, _ = create_adapter(model)

    with pytest.raises(EmbeddingDimensionError):
        await adapter.embed(
            EmbeddingRequest(text="hello")
        )


@pytest.mark.asyncio
async def test_non_numeric_vector_values_are_rejected() -> None:
    model = FakeSentenceTransformer(
        "fake-model",
        output=["not-a-number", 0.2, 0.3],
    )

    adapter, _ = create_adapter(model)

    with pytest.raises(EmbeddingDimensionError):
        await adapter.embed(
            EmbeddingRequest(text="hello")
        )


@pytest.mark.asyncio
async def test_batch_output_count_mismatch_is_rejected() -> None:
    model = FakeSentenceTransformer(
        "fake-model",
        output=[
            [0.1, 0.2, 0.3],
        ],
    )

    adapter, _ = create_adapter(model)

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="one"),
            EmbeddingRequest(text="two"),
        ]
    )

    with pytest.raises(EmbeddingDimensionError):
        await adapter.embed_batch(request)


@pytest.mark.asyncio
async def test_runtime_exception_is_translated_to_provider_error() -> None:
    model = FakeSentenceTransformer(
        "fake-model",
        encode_error=TimeoutError("model timed out"),
    )

    adapter, _ = create_adapter(model)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await adapter.embed(
            EmbeddingRequest(text="hello")
        )

    error = exc_info.value

    assert error.category == EmbeddingErrorCategory.TIMEOUT
    assert error.classification is not None
    assert error.classification.category == EmbeddingErrorCategory.TIMEOUT
    assert error.context.operation.value == "embed"
    assert error.context.input_count == 1


@pytest.mark.asyncio
async def test_batch_runtime_exception_contains_batch_context() -> None:
    model = FakeSentenceTransformer(
        "fake-model",
        encode_error=ConnectionError("network unavailable"),
    )

    adapter, _ = create_adapter(model)

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="one"),
            EmbeddingRequest(text="two"),
        ]
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await adapter.embed_batch(request)

    error = exc_info.value

    assert error.category == EmbeddingErrorCategory.NETWORK
    assert error.classification is not None
    assert error.classification.category == EmbeddingErrorCategory.NETWORK
    assert error.context.operation.value == "embed_batch"
    assert error.context.input_count == 2