from __future__ import annotations

import pytest

from app.models_ai.embeddings.adapters.sentence_transformer import (
    SentenceTransformerAdapter,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingModelConfig,
    EmbeddingRequest,
)


REAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


def create_real_adapter() -> SentenceTransformerAdapter:
    """Create a real Sentence Transformer adapter."""

    config = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name=REAL_MODEL,
        expected_dimension=EXPECTED_DIMENSION,
        normalized=True,
        batch_size=32,
        options={
            "device": "cpu",
        },
    )

    return SentenceTransformerAdapter(config)


@pytest.mark.asyncio
async def test_real_model_loads_lazily() -> None:
    adapter = create_real_adapter()

    assert adapter.is_loaded is False
    assert adapter.model is None

    model = adapter._load_model()

    assert model is not None
    assert adapter.is_loaded is True
    assert adapter.model is model


@pytest.mark.asyncio
async def test_real_single_embedding() -> None:
    adapter = create_real_adapter()

    request = EmbeddingRequest(
        text=(
            "Artificial intelligence is transforming "
            "modern software development."
        )
    )

    result = await adapter.embed(request)

    assert len(result.values) == EXPECTED_DIMENSION

    assert result.model.provider == "sentence-transformers"
    assert result.model.model_name == REAL_MODEL
    assert result.model.dimension == EXPECTED_DIMENSION
    assert result.model.normalized is True

    assert result.input_index == 0


@pytest.mark.asyncio
async def test_real_batch_embedding() -> None:
    adapter = create_real_adapter()

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(
                text=(
                    "Artificial intelligence is transforming "
                    "software development."
                )
            ),
            EmbeddingRequest(
                text=(
                    "Retrieval augmented generation improves "
                    "knowledge access."
                )
            ),
            EmbeddingRequest(
                text=(
                    "Vector embeddings represent semantic "
                    "information."
                )
            ),
        ]
    )

    result = await adapter.embed_batch(request)

    assert result.input_count == 3
    assert result.dimension == EXPECTED_DIMENSION
    assert len(result.embeddings) == 3

    for index, embedding in enumerate(result.embeddings):
        assert embedding.input_index == index
        assert len(embedding.values) == EXPECTED_DIMENSION
        assert embedding.model.dimension == EXPECTED_DIMENSION


@pytest.mark.asyncio
async def test_real_model_is_cached() -> None:
    adapter = create_real_adapter()

    first = adapter._load_model()
    second = adapter._load_model()

    assert first is second
    assert adapter.model is first


@pytest.mark.asyncio
async def test_real_model_uses_cpu_device() -> None:
    adapter = create_real_adapter()

    model = adapter._load_model()

    assert model is not None
    assert adapter.is_loaded is True


@pytest.mark.asyncio
async def test_real_embeddings_are_finite() -> None:
    adapter = create_real_adapter()

    result = await adapter.embed(
        EmbeddingRequest(
            text=(
                "This is a real Sentence Transformer "
                "inference test."
            )
        )
    )

    assert len(result.values) == EXPECTED_DIMENSION

    for value in result.values:
        assert value == value
        assert value not in (
            float("inf"),
            float("-inf"),
        )


@pytest.mark.asyncio
async def test_real_batch_preserves_input_order() -> None:
    adapter = create_real_adapter()

    texts = [
        "The first document discusses databases.",
        "The second document discusses neural networks.",
        "The third document discusses vector search.",
    ]

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text=text)
            for text in texts
        ]
    )

    result = await adapter.embed_batch(request)

    assert [
        embedding.input_index
        for embedding in result.embeddings
    ] == [0, 1, 2]

    assert len(result.embeddings) == len(texts)


@pytest.mark.asyncio
async def test_real_duplicate_inputs_remain_distinct() -> None:
    adapter = create_real_adapter()

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(
                text="duplicate input"
            ),
            EmbeddingRequest(
                text="duplicate input"
            ),
        ]
    )

    result = await adapter.embed_batch(request)

    assert result.input_count == 2
    assert len(result.embeddings) == 2

    assert result.embeddings[0].input_index == 0
    assert result.embeddings[1].input_index == 1

    assert (
        result.embeddings[0].values
        == result.embeddings[1].values
    )