from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingDimensionError,
    EmbeddingModelInfo,
    EmbeddingPort,
    EmbeddingProviderError,
    EmbeddingRequest,
    EmbeddingVector,
)


class FakeEmbeddingProvider(EmbeddingPort):
    """Deterministic provider used to test EmbeddingPort semantics."""

    def __init__(self) -> None:
        self.model = EmbeddingModelInfo(
            provider="fake",
            model_name="fake-embedding-model",
            dimension=3,
            normalized=True,
        )

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
        embeddings = [
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
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


class FailingEmbeddingProvider(EmbeddingPort):
    """Provider that simulates provider-level failures."""

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise EmbeddingProviderError(
            "Simulated embedding provider failure."
        )

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        raise EmbeddingBatchError(
            "Simulated batch embedding provider failure."
        )


class InconsistentDimensionProvider(EmbeddingPort):
    """Provider used to verify dimension consistency at the contract level."""

    def __init__(self) -> None:
        self.model = EmbeddingModelInfo(
            provider="fake",
            model_name="inconsistent-model",
            dimension=3,
        )

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
        embeddings = [
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=self.model,
                input_index=0,
            ),
            EmbeddingVector(
                values=[0.4, 0.5],
                model=self.model,
                input_index=1,
            ),
        ]

        raise EmbeddingDimensionError(
            "Embedding dimensions are inconsistent."
        )


@pytest.mark.asyncio
async def test_embed_has_one_to_one_semantics() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed(
        EmbeddingRequest(text="hello"),
    )

    assert isinstance(result, EmbeddingVector)
    assert result.input_index == 0
    assert len(result.values) == result.model.dimension


@pytest.mark.asyncio
async def test_embed_batch_returns_one_embedding_per_input() -> None:
    provider = FakeEmbeddingProvider()

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="first"),
            EmbeddingRequest(text="second"),
            EmbeddingRequest(text="third"),
            EmbeddingRequest(text="fourth"),
        ]
    )

    result = await provider.embed_batch(request)

    assert len(result.embeddings) == len(request.requests)
    assert result.input_count == len(request.requests)


@pytest.mark.asyncio
async def test_embed_batch_preserves_input_order() -> None:
    provider = FakeEmbeddingProvider()

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="first"),
            EmbeddingRequest(text="second"),
            EmbeddingRequest(text="third"),
        ]
    )

    result = await provider.embed_batch(request)

    assert [embedding.input_index for embedding in result.embeddings] == [
        0,
        1,
        2,
    ]


@pytest.mark.asyncio
async def test_embed_batch_preserves_duplicate_inputs_as_distinct_items() -> None:
    provider = FakeEmbeddingProvider()

    request = EmbeddingBatchRequest(
        requests=[
            EmbeddingRequest(text="duplicate"),
            EmbeddingRequest(text="duplicate"),
            EmbeddingRequest(text="different"),
        ]
    )

    result = await provider.embed_batch(request)

    assert len(result.embeddings) == 3
    assert [embedding.input_index for embedding in result.embeddings] == [
        0,
        1,
        2,
    ]


@pytest.mark.asyncio
async def test_embed_batch_uses_one_model_descriptor() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed_batch(
        EmbeddingBatchRequest(
            requests=[
                EmbeddingRequest(text="first"),
                EmbeddingRequest(text="second"),
            ]
        )
    )

    assert all(
        embedding.model == result.model
        for embedding in result.embeddings
    )


@pytest.mark.asyncio
async def test_embed_batch_uses_consistent_dimension() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed_batch(
        EmbeddingBatchRequest(
            requests=[
                EmbeddingRequest(text="first"),
                EmbeddingRequest(text="second"),
            ]
        )
    )

    assert result.dimension == result.model.dimension

    assert all(
        len(embedding.values) == result.dimension
        for embedding in result.embeddings
    )


@pytest.mark.asyncio
async def test_embed_batch_rejects_inconsistent_dimensions() -> None:
    provider = InconsistentDimensionProvider()

    with pytest.raises(EmbeddingDimensionError):
        await provider.embed_batch(
            EmbeddingBatchRequest(
                requests=[
                    EmbeddingRequest(text="first"),
                    EmbeddingRequest(text="second"),
                ]
            )
        )


@pytest.mark.asyncio
async def test_single_embedding_provider_failure_is_provider_independent() -> None:
    provider = FailingEmbeddingProvider()

    with pytest.raises(EmbeddingProviderError):
        await provider.embed(
            EmbeddingRequest(text="hello"),
        )


@pytest.mark.asyncio
async def test_batch_embedding_provider_failure_is_provider_independent() -> None:
    provider = FailingEmbeddingProvider()

    with pytest.raises(EmbeddingBatchError):
        await provider.embed_batch(
            EmbeddingBatchRequest(
                requests=[
                    EmbeddingRequest(text="first"),
                ]
            )
        )


@pytest.mark.asyncio
async def test_batch_result_indices_are_contiguous() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed_batch(
        EmbeddingBatchRequest(
            requests=[
                EmbeddingRequest(text="one"),
                EmbeddingRequest(text="two"),
                EmbeddingRequest(text="three"),
                EmbeddingRequest(text="four"),
                EmbeddingRequest(text="five"),
            ]
        )
    )

    assert [embedding.input_index for embedding in result.embeddings] == list(
        range(result.input_count)
    )


@pytest.mark.asyncio
async def test_batch_dimension_matches_every_embedding() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed_batch(
        EmbeddingBatchRequest(
            requests=[
                EmbeddingRequest(text="one"),
                EmbeddingRequest(text="two"),
            ]
        )
    )

    for embedding in result.embeddings:
        assert len(embedding.values) == result.dimension
        assert embedding.model.dimension == result.dimension


@pytest.mark.asyncio
async def test_single_embedding_does_not_require_batch_request() -> None:
    provider = FakeEmbeddingProvider()

    result = await provider.embed(
        EmbeddingRequest(text="single input"),
    )

    assert result.input_index == 0
    assert result.model.provider == "fake"


def test_embedding_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingPort()


def test_fake_provider_is_an_embedding_port() -> None:
    provider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingPort)