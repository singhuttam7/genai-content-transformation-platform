from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from app.models_ai.embeddings.schemas import (
    EmbeddingBatchResult,
    EmbeddingVector,
)
from app.vector_store.port import VectorStorePort
from app.vector_store.schemas import (
    VectorBatchPersistenceRequest,
    VectorDeleteRequest,
    VectorDeleteResult,
    VectorPersistenceRequest,
    VectorPersistenceResult,
    VectorRecord,
)


class VectorPersistenceService:
    """
    Application-level coordinator between generated embeddings
    and a vector persistence backend.

    This service deliberately contains no database-specific logic.
    Transaction ownership remains with the application layer.
    """

    def __init__(
        self,
        store: VectorStorePort,
    ) -> None:
        if not isinstance(store, VectorStorePort):
            raise TypeError(
                "store must implement VectorStorePort."
            )

        self._store = store

    @property
    def store(self) -> VectorStorePort:
        """Return the configured vector store port."""

        return self._store

    async def persist_embedding(
        self,
        *,
        chunk_id: UUID,
        embedding: EmbeddingVector,
        metadata: dict[str, object] | None = None,
    ) -> VectorRecord:
        """
        Persist one generated embedding for a knowledge chunk.
        """

        self._validate_chunk_id(chunk_id)

        if not isinstance(embedding, EmbeddingVector):
            raise TypeError(
                "embedding must be an EmbeddingVector."
            )

        record = self._to_record(
            chunk_id=chunk_id,
            embedding=embedding,
            metadata=metadata,
        )

        return await self._store.persist(
            VectorPersistenceRequest(
                chunk_id=chunk_id,
                embedding=record,
            )
        )

    async def persist_batch(
        self,
        *,
        chunk_ids: list[UUID],
        result: EmbeddingBatchResult,
        metadata: dict[str, object] | None = None,
    ) -> VectorPersistenceResult:
        """
        Persist a batch of generated embeddings.

        Input ordering is preserved between chunk_ids and embeddings.
        """

        if not chunk_ids:
            raise ValueError(
                "chunk_ids must contain at least one item."
            )

        if not isinstance(result, EmbeddingBatchResult):
            raise TypeError(
                "result must be an EmbeddingBatchResult."
            )

        if len(chunk_ids) != len(result.embeddings):
            raise ValueError(
                "chunk_ids count must match embedding count."
            )

        for chunk_id in chunk_ids:
            self._validate_chunk_id(chunk_id)

        records = [
            self._to_record(
                chunk_id=chunk_id,
                embedding=embedding,
                metadata=metadata,
            )
            for chunk_id, embedding in zip(
                chunk_ids,
                result.embeddings,
                strict=True,
            )
        ]

        return await self._store.persist_batch(
            VectorBatchPersistenceRequest(
                records=records,
            )
        )

    async def get_embedding(
        self,
        chunk_id: UUID,
    ) -> VectorRecord | None:
        """Retrieve the persisted vector for a knowledge chunk."""

        self._validate_chunk_id(chunk_id)

        return await self._store.get_by_chunk_id(
            chunk_id
        )

    async def exists(
        self,
        chunk_id: UUID,
    ) -> bool:
        """Return whether a persisted vector exists for a chunk."""

        self._validate_chunk_id(chunk_id)

        return await self._store.exists(
            chunk_id
        )

    async def delete(
        self,
        chunk_id: UUID,
    ) -> VectorDeleteResult:
        """Delete the persisted vector for a knowledge chunk."""

        self._validate_chunk_id(chunk_id)

        return await self._store.delete(
            VectorDeleteRequest(
                chunk_id=chunk_id,
            )
        )

    @staticmethod
    def _to_record(
        *,
        chunk_id: UUID,
        embedding: EmbeddingVector,
        metadata: dict[str, object] | None,
    ) -> VectorRecord:
        """
        Convert an embedding-domain object into a persistence
        contract while isolating mutable metadata.
        """

        combined_metadata: dict[str, object] = {}

        if metadata:
            combined_metadata.update(
                deepcopy(metadata)
            )

        combined_metadata["input_index"] = (
            embedding.input_index
        )

        return VectorRecord(
            chunk_id=chunk_id,
            values=list(embedding.values),
            model=embedding.model,
            metadata=combined_metadata,
        )

    @staticmethod
    def _validate_chunk_id(
        chunk_id: UUID,
    ) -> None:
        if not isinstance(chunk_id, UUID):
            raise TypeError(
                "chunk_id must be a UUID."
            )