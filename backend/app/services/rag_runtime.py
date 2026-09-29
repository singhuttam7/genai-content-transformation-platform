from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models_ai.embeddings.service import EmbeddingService
from app.rag.service import RAGRetrievalService
from app.services.embedding_runtime import create_embedding_service
from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)
from app.vector_store.retrieval.service import VectorRetrievalService


def create_rag_retrieval_service(
    session: AsyncSession,
    *,
    embedding_service: EmbeddingService | None = None,
) -> RAGRetrievalService:
    """
    Compose the application RAG retrieval service for one database session.

    The PostgreSQL retrieval adapter is intentionally session-scoped,
    while the embedding service may be supplied by the caller or
    created from the application's embedding runtime configuration.
    """

    if not isinstance(session, AsyncSession):
        raise TypeError(
            "session must be an AsyncSession.",
        )

    if embedding_service is None:
        embedding_service = create_embedding_service()

    if not isinstance(
        embedding_service,
        EmbeddingService,
    ):
        raise TypeError(
            "embedding_service must be an EmbeddingService.",
        )

    retrieval_port = PostgresVectorRetrieval(
        session,
    )

    retrieval_service = VectorRetrievalService(
        port=retrieval_port,
    )

    return RAGRetrievalService(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
    )