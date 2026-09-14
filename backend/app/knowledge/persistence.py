from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.chunk_service import KnowledgeChunkService
from app.knowledge.document_service import KnowledgeDocumentService
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_document import KnowledgeDocument


class KnowledgePersistenceService:
    """
    Coordinate atomic persistence of a knowledge document and its chunks.

    Transaction ownership remains with the caller. This service flushes
    changes but never commits or rolls back the session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

        self.documents = KnowledgeDocumentService(session)
        self.chunks = KnowledgeChunkService(session)

    async def persist_document_with_chunks(
        self,
        *,
        project_id: UUID,
        source_id: UUID,
        version: int,
        content_hash: str,
        chunks: list[dict],
        title: str | None = None,
        language: str | None = None,
        status: str = "COMPLETED",
        metadata: dict | None = None,
    ) -> tuple[KnowledgeDocument, list[KnowledgeChunk]]:
        """
        Persist a knowledge document and all of its chunks.

        The caller owns the transaction. If any operation fails, the
        caller is responsible for rollback.
        """

        document = await self.documents.create_document(
            project_id=project_id,
            source_id=source_id,
            version=version,
            content_hash=content_hash,
            title=title,
            language=language,
            status=status,
            metadata=metadata,
        )

        persisted_chunks = await self.chunks.create_chunks(
            document_id=document.id,
            chunks=chunks,
        )

        return document, persisted_chunks