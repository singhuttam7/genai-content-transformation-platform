from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_chunk_embedding import (
    KnowledgeChunkEmbedding,
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


class PostgresVectorStore(VectorStorePort):
    """
    PostgreSQL/pgvector implementation of VectorStorePort.

    This adapter owns database interaction only. It does not own
    transaction boundaries; callers are responsible for commit and
    rollback.
    """

    VECTOR_DIMENSION = 384

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        if not isinstance(session, AsyncSession):
            raise TypeError(
                "session must be an AsyncSession."
            )

        self._session = session

    @property
    def session(self) -> AsyncSession:
        """Return the configured database session."""

        return self._session

    async def persist(
        self,
        request: VectorPersistenceRequest,
    ) -> VectorRecord:
        """Persist one vector and flush the session."""

        record = request.embedding

        self._validate_record(record)

        entity = KnowledgeChunkEmbedding(
            chunk_id=record.chunk_id,
            provider=record.model.provider,
            model_name=record.model.model_name,
            dimension=record.model.dimension,
            normalized=record.model.normalized,
            embedding=list(record.values),
            vector_metadata=deepcopy(record.metadata),
        )

        self._session.add(entity)
        await self._session.flush()

        return record

    async def persist_batch(
        self,
        request: VectorBatchPersistenceRequest,
    ) -> VectorPersistenceResult:
        """Persist multiple vectors and flush once."""

        records = list(request.records)

        for record in records:
            self._validate_record(record)

        entities = [
            KnowledgeChunkEmbedding(
                chunk_id=record.chunk_id,
                provider=record.model.provider,
                model_name=record.model.model_name,
                dimension=record.model.dimension,
                normalized=record.model.normalized,
                embedding=list(record.values),
                vector_metadata=deepcopy(record.metadata),
            )
            for record in records
        ]

        self._session.add_all(entities)
        await self._session.flush()

        return VectorPersistenceResult(
            records=records,
            count=len(records),
        )

    async def get_by_chunk_id(
        self,
        chunk_id: UUID,
    ) -> VectorRecord | None:
        """Return the first persisted embedding for a chunk."""

        stmt = (
            select(KnowledgeChunkEmbedding)
            .where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
            .order_by(
                KnowledgeChunkEmbedding.created_at.asc()
            )
        )

        result = await self._session.execute(stmt)

        entity = result.scalars().first()

        if entity is None:
            return None

        return self._to_record(entity)

    async def exists(
        self,
        chunk_id: UUID,
    ) -> bool:
        """Return whether any embedding exists for a chunk."""

        stmt = (
            select(KnowledgeChunkEmbedding.id)
            .where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
            .limit(1)
        )

        result = await self._session.execute(stmt)

        return result.scalar_one_or_none() is not None

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> VectorDeleteResult:
        """Delete all persisted embeddings for a chunk."""

        stmt = delete(
            KnowledgeChunkEmbedding
        ).where(
            KnowledgeChunkEmbedding.chunk_id
            == request.chunk_id
        )

        result = await self._session.execute(stmt)

        deleted_count = result.rowcount or 0

        await self._session.flush()

        return VectorDeleteResult(
            chunk_id=request.chunk_id,
            deleted_count=deleted_count,
        )

    def _validate_record(
        self,
        record: VectorRecord,
    ) -> None:
        """Validate a record against the current pgvector schema."""

        dimension = len(record.values)

        if dimension != self.VECTOR_DIMENSION:
            raise ValueError(
                "Vector dimension must be "
                f"{self.VECTOR_DIMENSION}; got {dimension}."
            )

        if record.model.dimension != dimension:
            raise ValueError(
                "Embedding model dimension does not match "
                "vector length."
            )

        if record.model.dimension != self.VECTOR_DIMENSION:
            raise ValueError(
                "Embedding model dimension must be "
                f"{self.VECTOR_DIMENSION}; "
                f"got {record.model.dimension}."
            )

    @staticmethod
    def _to_record(
        entity: KnowledgeChunkEmbedding,
    ) -> VectorRecord:
        """Convert a database entity into the persistence contract."""

        from app.models_ai.embeddings.schemas import (
            EmbeddingModelInfo,
        )

        model = EmbeddingModelInfo(
            provider=entity.provider,
            model_name=entity.model_name,
            dimension=entity.dimension,
            normalized=entity.normalized,
        )

        return VectorRecord(
            chunk_id=entity.chunk_id,
            values=list(entity.embedding),
            model=model,
            metadata=deepcopy(
                entity.vector_metadata
            ),
        )