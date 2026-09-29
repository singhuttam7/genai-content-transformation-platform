from unittest.mock import MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.models_ai.embeddings.service import EmbeddingService
from app.rag.service import RAGRetrievalService
from app.services.rag_runtime import create_rag_retrieval_service
from app.vector_store.retrieval.service import VectorRetrievalService


def test_create_rag_retrieval_service():
    session = MagicMock(spec=AsyncSession)

    result = create_rag_retrieval_service(session)

    assert isinstance(result, RAGRetrievalService)
    assert isinstance(result.embedding_service, EmbeddingService)
    assert isinstance(result.retrieval_service, VectorRetrievalService)