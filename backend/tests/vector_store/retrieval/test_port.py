from uuid import uuid4

import pytest

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.retrieval.port import VectorRetrievalPort
from app.vector_store.retrieval.schemas import (
    VectorRetrievalRequest,
    VectorRetrievalResult,
)


def build_model_info() -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        normalized=True,
    )


class FakeVectorRetrievalPort(VectorRetrievalPort):
    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        return VectorRetrievalResult(
            matches=[],
            query_model=request.model,
            count=0,
        )


def test_retrieval_port_is_abstract():
    with pytest.raises(TypeError):
        VectorRetrievalPort()


@pytest.mark.asyncio
async def test_fake_retrieval_port_implements_contract():
    port = FakeVectorRetrievalPort()

    request = VectorRetrievalRequest(
        query_vector=[0.1] * 384,
        model=build_model_info(),
    )

    result = await port.search(request)

    assert isinstance(result, VectorRetrievalResult)
    assert result.count == 0
    assert result.query_model == request.model


@pytest.mark.asyncio
async def test_retrieval_port_preserves_query_model():
    port = FakeVectorRetrievalPort()

    model = build_model_info()

    request = VectorRetrievalRequest(
        query_vector=[0.0] * 384,
        model=model,
    )

    result = await port.search(request)

    assert result.query_model.provider == "sentence-transformers"
    assert (
        result.query_model.model_name
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert result.query_model.dimension == 384
    assert result.query_model.normalized is True


@pytest.mark.asyncio
async def test_retrieval_request_accepts_project_filter():
    port = FakeVectorRetrievalPort()

    project_id = uuid4()

    request = VectorRetrievalRequest(
        query_vector=[0.0] * 384,
        model=build_model_info(),
        project_id=project_id,
    )

    result = await port.search(request)

    assert result.count == 0