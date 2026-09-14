from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_chunk import KnowledgeChunk


class KnowledgeChunkService:
    """Persistence operations for knowledge chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_chunk(
        self,
        *,
        document_id: UUID,
        chunk_index: int,
        text: str,
        content_hash: str,
        token_count: int | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeChunk:
        """Create and flush a single knowledge chunk."""

        chunk = KnowledgeChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            content_hash=content_hash,
            token_count=token_count,
            chunk_metadata=metadata or {},
        )

        self.session.add(chunk)

        await self.session.flush()

        return chunk

    async def create_chunks(
        self,
        *,
        document_id: UUID,
        chunks: list[dict],
    ) -> list[KnowledgeChunk]:
        """
        Create multiple knowledge chunks atomically within the
        caller-owned transaction.
        """

        records: list[KnowledgeChunk] = []

        for chunk_data in chunks:
            chunk = KnowledgeChunk(
                document_id=document_id,
                chunk_index=chunk_data["chunk_index"],
                text=chunk_data["text"],
                content_hash=chunk_data["content_hash"],
                token_count=chunk_data.get("token_count"),
                chunk_metadata=chunk_data.get("metadata", {}),
            )

            self.session.add(chunk)
            records.append(chunk)

        await self.session.flush()

        return records

    async def get_by_id(
        self,
        *,
        chunk_id: UUID,
    ) -> KnowledgeChunk | None:
        """Return a knowledge chunk by primary key."""

        return await self.session.get(
            KnowledgeChunk,
            chunk_id,
        )

    async def list_for_document(
        self,
        *,
        document_id: UUID,
    ) -> list[KnowledgeChunk]:
        """Return all chunks for a document in deterministic order."""

        statement = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == document_id)
            .order_by(KnowledgeChunk.chunk_index.asc())
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def delete_for_document(
        self,
        *,
        document_id: UUID,
    ) -> int:
        """Delete all chunks belonging to a document."""

        statement = delete(KnowledgeChunk).where(
            KnowledgeChunk.document_id == document_id
        )

        result = await self.session.execute(statement)

        await self.session.flush()

        return result.rowcount or 0