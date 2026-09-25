from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.knowledge.provenance.schemas import KnowledgeChunkProvenance
from app.models_ai.embeddings.service import EmbeddingService
from app.vector_store.retrieval.schemas import (
    VectorRetrievalMatch,
    VectorRetrievalRequest,
)
from app.vector_store.retrieval.service import VectorRetrievalService

from app.rag.schemas import (
    RAGQuery,
    RAGRetrievedChunk,
)


class RAGRetrievalService:
    """
    Orchestrate query embedding and semantic knowledge retrieval.

    Responsibilities:
    - validate configured embedding and retrieval services;
    - embed the user's RAG query;
    - translate the embedding into a vector retrieval request;
    - execute semantic retrieval;
    - translate retrieval matches into RAGRetrievedChunk objects;
    - preserve retrieval metadata and valid provenance.

    Non-responsibilities:
    - chunking;
    - embedding model construction;
    - vector persistence;
    - context assembly;
    - prompt construction;
    - LLM invocation;
    - generation;
    - ranking beyond the configured vector retrieval service.
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        retrieval_service: VectorRetrievalService,
    ) -> None:
        if not isinstance(
            embedding_service,
            EmbeddingService,
        ):
            raise TypeError(
                "embedding_service must be an EmbeddingService."
            )

        if not isinstance(
            retrieval_service,
            VectorRetrievalService,
        ):
            raise TypeError(
                "retrieval_service must be a VectorRetrievalService."
            )

        self._embedding_service = embedding_service
        self._retrieval_service = retrieval_service

    @property
    def embedding_service(self) -> EmbeddingService:
        """Return the configured embedding service."""
        return self._embedding_service

    @property
    def retrieval_service(self) -> VectorRetrievalService:
        """Return the configured vector retrieval service."""
        return self._retrieval_service

    async def retrieve(
        self,
        query: RAGQuery,
    ) -> list[RAGRetrievedChunk]:
        """
        Embed a RAG query and retrieve matching knowledge chunks.

        The query's retrieval constraints are preserved exactly when
        constructing the vector retrieval request.

        Args:
            query:
                Validated application-level RAG query.

        Returns:
            Retrieved chunks ordered exactly as returned by the
            VectorRetrievalService.

        Raises:
            TypeError:
                If query is not a RAGQuery.
            ValueError:
                If the generated embedding is invalid.
            Exceptions from the embedding or retrieval services:
                Propagated without modification.
        """

        self._validate_query(query)

        embedding = await self._embedding_service.embed_text(
            query.text,
        )

        if not embedding.values:
            raise ValueError(
                "Query embedding must contain at least one value."
            )

        retrieval_request = VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=embedding.model,
            top_k=query.top_k,
            similarity_threshold=query.similarity_threshold,
            project_id=query.project_id,
            metadata_filter=deepcopy(
                query.metadata_filter,
            ),
        )

        retrieval_result = await self._retrieval_service.search(
            retrieval_request,
        )

        return [
            self._map_match(match)
            for match in retrieval_result.matches
        ]

    @staticmethod
    def _validate_query(
        query: RAGQuery,
    ) -> None:
        """Validate the application-level RAG query."""
        if not isinstance(query, RAGQuery):
            raise TypeError(
                "query must be a RAGQuery."
            )

    @classmethod
    def _map_match(
        cls,
        match: VectorRetrievalMatch,
    ) -> RAGRetrievedChunk:
        """
        Translate one vector retrieval match into a RAG chunk.

        Retrieval metadata remains intact. Provenance is extracted only
        when a valid KnowledgeChunkProvenance object is explicitly
        present in the retrieval metadata.
        """

        if not isinstance(
            match,
            VectorRetrievalMatch,
        ):
            raise TypeError(
                "match must be a VectorRetrievalMatch."
            )

        metadata = deepcopy(match.metadata)

        provenance = cls._extract_provenance(
            metadata,
        )

        return RAGRetrievedChunk(
            chunk_id=match.chunk_id,
            text=match.text,
            similarity=match.similarity,
            model=match.model,
            provenance=provenance,
            metadata=metadata,
        )

    @staticmethod
    def _extract_provenance(
        metadata: dict[str, Any],
    ) -> KnowledgeChunkProvenance | None:
        """
        Extract valid chunk provenance from retrieval metadata.

        The retrieval layer may expose JSON-compatible provenance
        metadata. Invalid or absent provenance is intentionally treated
        as unavailable rather than being fabricated.
        """

        value = metadata.get("provenance")

        if value is None:
            return None

        if isinstance(
            value,
            KnowledgeChunkProvenance,
        ):
            return value

        if not isinstance(value, dict):
            return None

        try:
            return KnowledgeChunkProvenance.model_validate(
                value,
            )
        except Exception:
            return None