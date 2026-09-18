from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding
from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.retrieval.port import VectorRetrievalPort
from app.vector_store.retrieval.schemas import (
    VectorRetrievalMatch,
    VectorRetrievalRequest,
    VectorRetrievalResult,
)


class PostgresVectorRetrieval(VectorRetrievalPort):
    """PostgreSQL/pgvector implementation of vector retrieval."""

    VECTOR_DIMENSION = 384

    def __init__(self, session: AsyncSession) -> None:
        if not isinstance(session, AsyncSession):
            raise TypeError("session must be an AsyncSession.")

        self._session = session

    @property
    def session(self) -> AsyncSession:
        """Return the underlying database session."""
        return self._session

    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        """
        Search knowledge chunk embeddings using cosine similarity.

        Retrieval is isolated by embedding provider, model name,
        and vector dimension.
        """
        self._validate_request(request)

        query_vector = self._validate_query_dimension(
            request.query_vector,
            request.model.dimension,
        )

        distance = KnowledgeChunkEmbedding.embedding.cosine_distance(
            query_vector
        )

        similarity = (1.0 - distance).label("similarity")

        statement = (
            select(
                KnowledgeChunkEmbedding,
                KnowledgeChunk,
                similarity,
            )
            .join(
                KnowledgeChunk,
                KnowledgeChunk.id
                == KnowledgeChunkEmbedding.chunk_id,
            )
            .join(
                KnowledgeChunk.document,
            )
            .where(
                and_(
                    KnowledgeChunkEmbedding.provider
                    == request.model.provider,
                    KnowledgeChunkEmbedding.model_name
                    == request.model.model_name,
                    KnowledgeChunkEmbedding.dimension
                    == request.model.dimension,
                )
            )
            .order_by(
                distance.asc(),
            )
            .limit(
                request.top_k,
            )
        )

        if request.project_id is not None:
            statement = statement.where(
                KnowledgeChunk.document.has(
                    project_id=request.project_id,
                )
            )

        if request.metadata_filter:
            statement = statement.where(
                KnowledgeChunk.chunk_metadata.contains(
                    deepcopy(request.metadata_filter),
                )
            )

        if request.similarity_threshold is not None:
            statement = statement.where(
                similarity >= request.similarity_threshold,
            )

        result = await self._session.execute(statement)

        rows = result.all()

        matches = [
            self._build_match(
                embedding=embedding,
                chunk=chunk,
                similarity_score=similarity_score,
            )
            for embedding, chunk, similarity_score in rows
        ]

        return VectorRetrievalResult(
            matches=matches,
            query_model=request.model,
            count=len(matches),
            metadata={
                "provider": request.model.provider,
                "model_name": request.model.model_name,
                "top_k": request.top_k,
                "similarity_metric": "cosine",
            },
        )

    def _validate_request(
        self,
        request: VectorRetrievalRequest,
    ) -> None:
        """Validate the retrieval request."""
        if not isinstance(request, VectorRetrievalRequest):
            raise TypeError(
                "request must be a VectorRetrievalRequest."
            )

        if request.model.dimension != self.VECTOR_DIMENSION:
            raise ValueError(
                "Unsupported embedding dimension: "
                f"{request.model.dimension}. "
                f"Expected {self.VECTOR_DIMENSION}."
            )

    def _validate_query_dimension(
        self,
        query_vector: list[float],
        expected_dimension: int,
    ) -> list[float]:
        """Validate query vector dimensions."""
        if len(query_vector) != expected_dimension:
            raise ValueError(
                "Query vector dimension does not match "
                "the embedding model dimension."
            )

        if len(query_vector) != self.VECTOR_DIMENSION:
            raise ValueError(
                "Query vector dimension does not match "
                "the PostgreSQL vector dimension."
            )

        return list(query_vector)

    def _get_chunk_metadata(
        self,
        chunk: KnowledgeChunk,
    ) -> dict[str, Any]:
        """
        Return the persisted JSONB metadata associated with a chunk.

        The KnowledgeChunk model maps the database column named
        ``metadata`` to the Python attribute ``chunk_metadata``.
        """
        value = chunk.chunk_metadata

        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError(
                "KnowledgeChunk.chunk_metadata must be a dictionary."
            )

        return deepcopy(value)

    def _build_match(
        self,
        *,
        embedding: KnowledgeChunkEmbedding,
        chunk: KnowledgeChunk,
        similarity_score: float,
    ) -> VectorRetrievalMatch:
        """
        Convert persisted embedding/chunk data into a retrieval match.

        The persistence model stores model information as individual
        columns. The retrieval contract expects an EmbeddingModelInfo,
        so the adapter reconstructs that contract here.
        """
        metadata = self._get_chunk_metadata(chunk)

        vector_metadata = getattr(
            embedding,
            "vector_metadata",
            None,
        )

        if vector_metadata:
            metadata.setdefault(
                "embedding",
                deepcopy(vector_metadata),
            )

        model = EmbeddingModelInfo(
            provider=embedding.provider,
            model_name=embedding.model_name,
            dimension=embedding.dimension,
            normalized=embedding.normalized,
        )

        return VectorRetrievalMatch(
            chunk_id=chunk.id,
            similarity=float(similarity_score),
            model=model,
            metadata=metadata,
        )