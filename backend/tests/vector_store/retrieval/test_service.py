from __future__ import annotations

from uuid import uuid4

import pytest

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.retrieval.port import VectorRetrievalPort
from app.vector_store.retrieval.schemas import (
    VectorRetrievalMatch,
    VectorRetrievalRequest,
    VectorRetrievalResult,
)
from app.vector_store.retrieval.service import VectorRetrievalService


VECTOR_DIMENSION = 384


def make_model() -> EmbeddingModelInfo:
    """Create a deterministic test embedding model."""
    return EmbeddingModelInfo(
        provider="test-provider",
        model_name="test-model",
        dimension=VECTOR_DIMENSION,
        normalized=True,
    )


def make_request(
    *,
    vector: list[float] | None = None,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    project_id=None,
    metadata_filter: dict | None = None,
) -> VectorRetrievalRequest:
    """Create a deterministic retrieval request."""
    return VectorRetrievalRequest(
        query_vector=vector or [1.0] * VECTOR_DIMENSION,
        model=make_model(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        project_id=project_id,
        metadata_filter=metadata_filter or {},
    )


def make_result(
    request: VectorRetrievalRequest,
) -> VectorRetrievalResult:
    """Create a deterministic retrieval result."""
    return VectorRetrievalResult(
        matches=[
            VectorRetrievalMatch(
                chunk_id=uuid4(),
                similarity=0.95,
                model=request.model,
                metadata={"section": "test"},
            )
        ],
        query_model=request.model,
        count=1,
        metadata={
            "provider": request.model.provider,
            "model_name": request.model.model_name,
            "top_k": request.top_k,
            "similarity_metric": "cosine",
        },
    )


class FakeVectorRetrievalPort(VectorRetrievalPort):
    """Fake retrieval port for service-level tests."""

    def __init__(
        self,
        result: VectorRetrievalResult | None = None,
    ) -> None:
        self.result = result
        self.requests: list[VectorRetrievalRequest] = []

    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        self.requests.append(request)

        if self.result is not None:
            return self.result

        return make_result(request)


def test_service_requires_retrieval_port() -> None:
    """The service must reject invalid port implementations."""
    with pytest.raises(
        TypeError,
        match="port must be a VectorRetrievalPort",
    ):
        VectorRetrievalService(object())  # type: ignore[arg-type]


def test_service_exposes_configured_port() -> None:
    """The configured retrieval port must be exposed."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    assert service.port is port


@pytest.mark.asyncio
async def test_search_delegates_to_port() -> None:
    """Search must delegate valid requests to the retrieval port."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)
    request = make_request()

    result = await service.search(request)

    assert len(port.requests) == 1
    assert port.requests[0] == request
    assert result.count == 1


@pytest.mark.asyncio
async def test_search_returns_port_result_unchanged() -> None:
    """The service must preserve the result returned by the port."""
    request = make_request()
    expected = make_result(request)

    port = FakeVectorRetrievalPort(result=expected)
    service = VectorRetrievalService(port)

    result = await service.search(request)

    assert result is expected


@pytest.mark.asyncio
async def test_search_preserves_top_k() -> None:
    """The service must preserve the requested top_k."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)
    request = make_request(top_k=10)

    await service.search(request)

    assert port.requests[0].top_k == 10


@pytest.mark.asyncio
async def test_search_preserves_similarity_threshold() -> None:
    """The service must preserve the similarity threshold."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)
    request = make_request(similarity_threshold=0.75)

    await service.search(request)

    assert port.requests[0].similarity_threshold == 0.75


@pytest.mark.asyncio
async def test_search_preserves_project_filter() -> None:
    """The service must preserve project isolation."""
    project_id = uuid4()

    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)
    request = make_request(project_id=project_id)

    await service.search(request)

    assert port.requests[0].project_id == project_id


@pytest.mark.asyncio
async def test_search_preserves_metadata_filter() -> None:
    """The service must preserve metadata filtering."""
    metadata_filter = {
        "section": "Introduction",
        "page": 5,
    }

    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)
    request = make_request(
        metadata_filter=metadata_filter,
    )

    await service.search(request)

    assert port.requests[0].metadata_filter == metadata_filter


@pytest.mark.asyncio
async def test_search_rejects_invalid_request_type() -> None:
    """The service must reject non-retrieval request objects."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    with pytest.raises(
        TypeError,
        match="request must be a VectorRetrievalRequest",
    ):
        await service.search(object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_search_rejects_empty_query_vector() -> None:
    """The service must reject an empty query vector."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request()
    object.__setattr__(request, "query_vector", [])

    with pytest.raises(
        ValueError,
        match="query_vector must not be empty",
    ):
        await service.search(request)

    assert port.requests == []


@pytest.mark.asyncio
async def test_search_rejects_invalid_top_k() -> None:
    """The service must reject non-positive top_k."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request()
    object.__setattr__(request, "top_k", 0)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than or equal to 1",
    ):
        await service.search(request)

    assert port.requests == []


@pytest.mark.asyncio
async def test_search_rejects_invalid_similarity_threshold() -> None:
    """The service must reject thresholds outside the valid range."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request()
    object.__setattr__(request, "similarity_threshold", 1.1)

    with pytest.raises(
        ValueError,
        match="similarity_threshold must be between",
    ):
        await service.search(request)

    assert port.requests == []


@pytest.mark.asyncio
async def test_search_rejects_non_numeric_query_vector() -> None:
    """The service must reject non-numeric query vector values."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request()
    object.__setattr__(
        request,
        "query_vector",
        ["invalid"] * VECTOR_DIMENSION,
    )

    with pytest.raises(
        ValueError,
        match="query_vector must contain only numeric values",
    ):
        await service.search(request)

    assert port.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
async def test_search_rejects_non_finite_query_vector(
    value: float,
) -> None:
    """The service must reject NaN and infinite vector values."""
    port = FakeVectorRetrievalPort()
    service = VectorRetrievalService(port)

    request = make_request()
    vector = [1.0] * VECTOR_DIMENSION
    vector[0] = value
    object.__setattr__(
        request,
        "query_vector",
        vector,
    )

    with pytest.raises(
        ValueError,
        match="query_vector must contain only finite numeric values",
    ):
        await service.search(request)

    assert port.requests == []


@pytest.mark.asyncio
async def test_search_propagates_port_failure() -> None:
    """Failures from the retrieval port must propagate unchanged."""

    class FailingPort(VectorRetrievalPort):
        async def search(
            self,
            request: VectorRetrievalRequest,
        ) -> VectorRetrievalResult:
            raise RuntimeError("retrieval backend unavailable")

    service = VectorRetrievalService(FailingPort())

    with pytest.raises(
        RuntimeError,
        match="retrieval backend unavailable",
    ):
        await service.search(make_request())