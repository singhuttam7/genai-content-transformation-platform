from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.vector_store.schemas import (
    VectorBatchPersistenceRequest,
    VectorDeleteRequest,
    VectorDeleteResult,
    VectorPersistenceRequest,
    VectorPersistenceResult,
    VectorRecord,
)


class VectorStorePort(ABC):
    """
    Provider-independent port for vector persistence.

    Implementations may use PostgreSQL/pgvector, Qdrant, Milvus,
    or another vector backend.

    Transaction ownership remains outside this port.
    """

    @abstractmethod
    async def persist(
        self,
        request: VectorPersistenceRequest,
    ) -> VectorRecord:
        """Persist one vector record."""

        raise NotImplementedError

    @abstractmethod
    async def persist_batch(
        self,
        request: VectorBatchPersistenceRequest,
    ) -> VectorPersistenceResult:
        """Persist multiple vector records."""

        raise NotImplementedError

    @abstractmethod
    async def get_by_chunk_id(
        self,
        chunk_id: UUID,
    ) -> VectorRecord | None:
        """Return the persisted vector for a chunk, if present."""

        raise NotImplementedError

    @abstractmethod
    async def exists(
        self,
        chunk_id: UUID,
    ) -> bool:
        """Return whether a vector exists for a chunk."""

        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> VectorDeleteResult:
        """Delete persisted vectors associated with a chunk."""

        raise NotImplementedError