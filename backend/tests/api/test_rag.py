from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.api.dependencies import get_rag_retrieval_service
from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.rag.schemas import (
    RAGQuery,
    RAGRetrievedChunk,
)
from app.rag.service import RAGRetrievalService


class FakeRAGRetrievalService(RAGRetrievalService):
    def __init__(self) -> None:
        self.received_query: RAGQuery | None = None

    async def retrieve(
        self,
        query: RAGQuery,
    ) -> list[RAGRetrievedChunk]:
        self.received_query = query

        return [
            RAGRetrievedChunk(
                chunk_id=uuid4(),
                text="Retrieval searches the knowledge base.",
                similarity=0.91,
                model=EmbeddingModelInfo(
                    provider="test",
                    model_name="test-model",
                    dimension=3,
                    normalized=True,
                ),
                metadata={
                    "source": "test",
                },
            ),
        ]


@pytest.fixture
def fake_rag_service():
    service = FakeRAGRetrievalService()

    app.dependency_overrides[
        get_rag_retrieval_service
    ] = lambda: service

    yield service

    app.dependency_overrides.pop(
        get_rag_retrieval_service,
        None,
    )


@pytest.mark.asyncio
async def test_rag_query_returns_context(
    fake_rag_service,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={
                "text": "How does retrieval work?",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["query"]["text"] == (
        "How does retrieval work?"
    )
    assert len(body["chunks"]) == 1
    assert (
        body["chunks"][0]["text"]
        == "Retrieval searches the knowledge base."
    )
    assert (
        body["chunks"][0]["similarity"]
        == 0.91
    )
    assert (
        "Retrieved Knowledge Context"
        in body["context_text"]
    )

    assert fake_rag_service.received_query is not None
    assert (
        fake_rag_service.received_query.text
        == "How does retrieval work?"
    )


@pytest.mark.asyncio
async def test_rag_query_rejects_blank_text(
    fake_rag_service,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={
                "text": "   ",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_rejects_invalid_top_k(
    fake_rag_service,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={
                "text": "retrieval",
                "top_k": 0,
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_rejects_unknown_fields(
    fake_rag_service,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={
                "text": "retrieval",
                "unexpected": True,
            },
        )

    assert response.status_code == 422