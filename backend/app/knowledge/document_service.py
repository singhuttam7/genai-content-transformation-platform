from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_document import KnowledgeDocument


class KnowledgeDocumentService:
    """Persistence operations for knowledge documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_document(
        self,
        *,
        project_id: UUID,
        source_id: UUID,
        version: int,
        content_hash: str,
        title: str | None = None,
        language: str | None = None,
        status: str = "PENDING",
        metadata: dict | None = None,
    ) -> KnowledgeDocument:
        """Create and flush a new knowledge document."""

        document = KnowledgeDocument(
            project_id=project_id,
            source_id=source_id,
            version=version,
            title=title,
            language=language,
            content_hash=content_hash,
            status=status,
            document_metadata=metadata or {},
        )

        self.session.add(document)

        await self.session.flush()

        return document

    async def get_by_id(
        self,
        *,
        document_id: UUID,
    ) -> KnowledgeDocument | None:
        """Return a knowledge document by primary key."""

        return await self.session.get(
            KnowledgeDocument,
            document_id,
        )

    async def get_by_source_version(
        self,
        *,
        source_id: UUID,
        version: int,
    ) -> KnowledgeDocument | None:
        """Return a specific source/version knowledge document."""

        statement = select(KnowledgeDocument).where(
            KnowledgeDocument.source_id == source_id,
            KnowledgeDocument.version == version,
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_latest_for_source(
        self,
        *,
        source_id: UUID,
    ) -> KnowledgeDocument | None:
        """Return the highest-version knowledge document for a source."""

        statement = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.source_id == source_id)
            .order_by(KnowledgeDocument.version.desc())
            .limit(1)
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_by_content_hash(
        self,
        *,
        source_id: UUID,
        content_hash: str,
    ) -> KnowledgeDocument | None:
        """Return a source document matching the supplied content hash."""

        statement = (
            select(KnowledgeDocument)
            .where(
                KnowledgeDocument.source_id == source_id,
                KnowledgeDocument.content_hash == content_hash,
            )
            .order_by(KnowledgeDocument.version.desc())
            .limit(1)
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def list_for_project(
        self,
        *,
        project_id: UUID,
    ) -> list[KnowledgeDocument]:
        """Return knowledge documents belonging to a project."""

        statement = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.project_id == project_id)
            .order_by(
                KnowledgeDocument.created_at.desc(),
                KnowledgeDocument.version.desc(),
            )
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def update_status(
        self,
        *,
        document: KnowledgeDocument,
        status: str,
    ) -> KnowledgeDocument:
        """Update document lifecycle status."""

        document.status = status

        await self.session.flush()

        return document