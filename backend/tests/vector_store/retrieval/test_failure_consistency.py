from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.retrieval.port import VectorRetrievalPort
from app.vector_store.retrieval.schemas import (
    VectorRetrievalMatch,
    VectorRetrievalRequest,
    VectorRetrievalResult,
)
from app.vector_store.retrieval.service import VectorRetrievalService


VECTOR_DIMENSION = 384


def make_model(
    provider: str = "failure-provider",
    model_name: str = "failure-model",
) -> EmbeddingModelInfo:
    """Create a deterministic embedding model."""
    return EmbeddingModelInfo(
        provider=provider,
        model_name=model_name,
        dimension=VECTOR_DIMENSION,
        normalized=True,
    )


def make_vector(
    dimension: int = VECTOR_DIMENSION,
) -> list[float]:
    """Create a deterministic vector."""
    return [1.0] + [0.0] * (dimension - 1)


def make_request(
    *,
    vector: list[float] | None = None,
    model: EmbeddingModelInfo | None = None,
    top_k: int = 5,
    similarity_threshold: float | None = None,
) -> VectorRetrievalRequest:
    """Create a retrieval request."""
    return VectorRetrievalRequest(
        query_vector=vector or make_vector(),
        model=model or make_model(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


class SuccessfulRetrievalPort(VectorRetrievalPort):
    """Minimal successful retrieval port."""

    def __init__(self) -> None:
        self.called = False
        self.last_request: VectorRetrievalRequest | None = None

    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        self.called = True
        self.last_request = request

        match = VectorRetrievalMatch(
            chunk_id=uuid4(),
            text="Retrieved test knowledge content.",
            similarity=1.0,
            model=request.model,
            metadata={},
        )

        return VectorRetrievalResult(
            matches=[match],
            query_model=request.model,
            count=1,
            metadata={},
        )


class FailingRetrievalPort(VectorRetrievalPort):
    """Retrieval port that deliberately fails."""

    def __init__(
        self,
        error: Exception,
    ) -> None:
        self.error = error
        self.called = False

    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        self.called = True
        raise self.error


@pytest.mark.asyncio
async def test_service_rejects_query_vector_dimension_below_model_dimension() -> None:
    """A query vector shorter than the model dimension must fail."""
    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request(
        vector=make_vector(VECTOR_DIMENSION - 1),
    )

    with pytest.raises(
        ValueError,
        match="query_vector dimension must match model dimension",
    ):
        await service.search(request)

    assert port.called is False


@pytest.mark.asyncio
async def test_service_rejects_query_vector_dimension_above_model_dimension() -> None:
    """A query vector longer than the model dimension must fail."""
    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request(
        vector=make_vector(VECTOR_DIMENSION + 1),
    )

    with pytest.raises(
        ValueError,
        match="query_vector dimension must match model dimension",
    ):
        await service.search(request)

    assert port.called is False


@pytest.mark.asyncio
async def test_service_rejects_nan_query_vector_value() -> None:
    """NaN must never reach the retrieval port."""
    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    vector = make_vector()
    vector[10] = float("nan")

    request = make_request(vector=vector)

    with pytest.raises(
        ValueError,
        match="finite numeric values",
    ):
        await service.search(request)

    assert port.called is False


@pytest.mark.asyncio
async def test_service_rejects_positive_infinity_query_vector_value() -> None:
    """Positive infinity must never reach the retrieval port."""
    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    vector = make_vector()
    vector[10] = float("inf")

    request = make_request(vector=vector)

    with pytest.raises(
        ValueError,
        match="finite numeric values",
    ):
        await service.search(request)

    assert port.called is False


@pytest.mark.asyncio
async def test_service_rejects_negative_infinity_query_vector_value() -> None:
    """Negative infinity must never reach the retrieval port."""
    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    vector = make_vector()
    vector[10] = float("-inf")

    request = make_request(vector=vector)

    with pytest.raises(
        ValueError,
        match="finite numeric values",
    ):
        await service.search(request)

    assert port.called is False


def test_request_rejects_invalid_top_k() -> None:
    """VectorRetrievalRequest must reject non-positive top_k."""
    with pytest.raises(ValidationError):
        make_request(top_k=0)


def test_request_rejects_similarity_threshold_above_one() -> None:
    """VectorRetrievalRequest must reject threshold values above one."""
    with pytest.raises(ValidationError):
        make_request(similarity_threshold=1.01)


def test_request_rejects_similarity_threshold_below_minus_one() -> None:
    """VectorRetrievalRequest must reject threshold values below minus one."""
    with pytest.raises(ValidationError):
        make_request(similarity_threshold=-1.01)


@pytest.mark.asyncio
async def test_infrastructure_failure_is_propagated() -> None:
    """Infrastructure failures must not become empty retrieval results."""
    error = RuntimeError("database connection failed")

    port = FailingRetrievalPort(error)
    service = VectorRetrievalService(port)

    request = make_request()

    with pytest.raises(
        RuntimeError,
        match="database connection failed",
    ):
        await service.search(request)

    assert port.called is True


@pytest.mark.asyncio
async def test_retrieval_result_count_matches_match_count() -> None:
    """A successful result must maintain count consistency."""
    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request()

    result = await service.search(request)

    assert result.count == len(result.matches)


@pytest.mark.asyncio
async def test_retrieval_result_preserves_requested_model() -> None:
    """The result must preserve the requested model identity."""
    model = make_model(
        provider="specific-provider",
        model_name="specific-model",
    )

    port = SuccessfulRetrievalPort()
    service = VectorRetrievalService(port)

    result = await service.search(
        make_request(model=model),
    )

    assert result.query_model == model
    assert result.matches[0].model == model