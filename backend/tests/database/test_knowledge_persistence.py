from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.session import SessionFactory
from app.knowledge.chunk_service import KnowledgeChunkService
from app.knowledge.document_service import KnowledgeDocumentService
from app.knowledge.persistence import KnowledgePersistenceService
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_document import KnowledgeDocument
from app.models.project import Project
from app.models.source import Source
from app.models.user import User


# ============================================================
# Test Helpers
# ============================================================


async def create_test_project() -> tuple[UUID, UUID]:
    """Create a temporary user and project."""

    user_id = uuid4()
    project_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"knowledge-test-{user_id}@example.com",
            name="Knowledge Persistence Test User",
        )

        project = Project(
            id=project_id,
            name=f"Knowledge Test Project {project_id}",
            owner_id=user_id,
        )

        session.add(user)
        session.add(project)

        await session.commit()

    return user_id, project_id


async def create_test_source(
    project_id: UUID,
    *,
    content_hash: str = "source-hash-001",
) -> UUID:
    """Create a temporary source belonging to a test project."""

    source_id = uuid4()

    async with SessionFactory() as session:
        source = Source(
            id=source_id,
            project_id=project_id,
            source_type="text",
            title="Knowledge Test Source",
            original_filename="knowledge-test.txt",
            mime_type="text/plain",
            storage_uri="file:///tmp/knowledge-test.txt",
            content_hash=content_hash,
            source_metadata={
                "test": True,
            },
            status="COMPLETED",
        )

        session.add(source)

        await session.commit()

    return source_id


async def cleanup_test_project(
    user_id: UUID,
    project_id: UUID,
) -> None:
    """Remove all test records."""

    async with SessionFactory() as session:
        project = await session.get(
            Project,
            project_id,
        )

        user = await session.get(
            User,
            user_id,
        )

        if project is not None:
            await session.delete(project)

        if user is not None:
            await session.delete(user)

        await session.commit()


# ============================================================
# Knowledge Document Tests
# ============================================================


@pytest.mark.asyncio
async def test_knowledge_document_persistence() -> None:
    """Verify complete KnowledgeDocument persistence lifecycle."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    document_id = None

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            document = await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                title="Test Knowledge Document",
                language="en",
                content_hash="document-hash-001",
                status="COMPLETED",
                metadata={
                    "test": True,
                    "document_type": "article",
                },
            )

            document_id = document.id

            assert document.id is not None
            assert document.project_id == project_id
            assert document.source_id == source_id
            assert document.version == 1
            assert document.title == "Test Knowledge Document"
            assert document.language == "en"
            assert document.content_hash == "document-hash-001"
            assert document.status == "COMPLETED"
            assert document.document_metadata["test"] is True
            assert (
                document.document_metadata["document_type"]
                == "article"
            )

            await session.commit()

        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            stored = await service.get_by_id(
                document_id=document_id,
            )

            assert stored is not None
            assert stored.id == document_id
            assert stored.project_id == project_id
            assert stored.source_id == source_id
            assert stored.version == 1
            assert stored.content_hash == "document-hash-001"

        print(
            "KnowledgeDocument persistence: OK"
        )

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_knowledge_document_source_version_lookup() -> None:
    """Verify source/version lookup."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            document = await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=3,
                content_hash="version-3-hash",
            )

            await session.commit()

            found = await service.get_by_source_version(
                source_id=source_id,
                version=3,
            )

            assert found is not None
            assert found.id == document.id
            assert found.version == 3

            missing = await service.get_by_source_version(
                source_id=source_id,
                version=99,
            )

            assert missing is None

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_knowledge_document_latest_version() -> None:
    """Verify latest document resolution."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="hash-v1",
            )

            await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=2,
                content_hash="hash-v2",
            )

            await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=5,
                content_hash="hash-v5",
            )

            await session.commit()

            latest = await service.get_latest_for_source(
                source_id=source_id,
            )

            assert latest is not None
            assert latest.version == 5
            assert latest.content_hash == "hash-v5"

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_knowledge_document_content_hash_lookup() -> None:
    """Verify content-hash lookup."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            document = await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="stable-content-hash",
            )

            await session.commit()

            found = await service.get_by_content_hash(
                source_id=source_id,
                content_hash="stable-content-hash",
            )

            assert found is not None
            assert found.id == document.id

            missing = await service.get_by_content_hash(
                source_id=source_id,
                content_hash="does-not-exist",
            )

            assert missing is None

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_knowledge_document_project_isolation() -> None:
    """Verify project-level document isolation."""

    user_1, project_1 = await create_test_project()
    user_2, project_2 = await create_test_project()

    source_1 = await create_test_source(
        project_1,
    )

    source_2 = await create_test_source(
        project_2,
        content_hash="source-hash-002",
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            document_1 = await service.create_document(
                project_id=project_1,
                source_id=source_1,
                version=1,
                content_hash="project-1-document",
            )

            document_2 = await service.create_document(
                project_id=project_2,
                source_id=source_2,
                version=1,
                content_hash="project-2-document",
            )

            await session.commit()

            project_1_documents = (
                await service.list_for_project(
                    project_id=project_1,
                )
            )

            project_2_documents = (
                await service.list_for_project(
                    project_id=project_2,
                )
            )

            assert len(project_1_documents) == 1
            assert len(project_2_documents) == 1

            assert project_1_documents[0].id == document_1.id
            assert project_2_documents[0].id == document_2.id

            assert all(
                document.project_id == project_1
                for document in project_1_documents
            )

            assert all(
                document.project_id == project_2
                for document in project_2_documents
            )

    finally:
        await cleanup_test_project(
            user_1,
            project_1,
        )

        await cleanup_test_project(
            user_2,
            project_2,
        )


@pytest.mark.asyncio
async def test_knowledge_document_status_update() -> None:
    """Verify lifecycle status update."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            document = await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="status-test-hash",
                status="PENDING",
            )

            await service.update_status(
                document=document,
                status="COMPLETED",
            )

            await session.commit()

        async with SessionFactory() as session:
            stored = await session.get(
                KnowledgeDocument,
                document.id,
            )

            assert stored is not None
            assert stored.status == "COMPLETED"

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_duplicate_source_version_is_rejected() -> None:
    """
    Verify the source/version uniqueness constraint.

    KnowledgeDocumentService.flush() executes the INSERT before
    the caller commits, so IntegrityError is expected during
    create_document().
    """

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            await service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="first-document",
            )

            await session.commit()

        async with SessionFactory() as session:
            service = KnowledgeDocumentService(
                session,
            )

            with pytest.raises(IntegrityError):
                await service.create_document(
                    project_id=project_id,
                    source_id=source_id,
                    version=1,
                    content_hash="duplicate-document",
                )

            await session.rollback()

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


# ============================================================
# Knowledge Chunk Tests
# ============================================================


@pytest.mark.asyncio
async def test_knowledge_chunk_persistence_and_ordering() -> None:
    """Verify chunk persistence and deterministic ordering."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            document_service = KnowledgeDocumentService(
                session,
            )

            chunk_service = KnowledgeChunkService(
                session,
            )

            document = await document_service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="chunk-document-hash",
            )

            chunks = await chunk_service.create_chunks(
                document_id=document.id,
                chunks=[
                    {
                        "chunk_index": 2,
                        "text": "Third chunk",
                        "content_hash": "chunk-2",
                        "token_count": 2,
                        "metadata": {
                            "section": "body",
                        },
                    },
                    {
                        "chunk_index": 0,
                        "text": "First chunk",
                        "content_hash": "chunk-0",
                        "token_count": 2,
                        "metadata": {
                            "section": "introduction",
                        },
                    },
                    {
                        "chunk_index": 1,
                        "text": "Second chunk",
                        "content_hash": "chunk-1",
                        "token_count": 2,
                        "metadata": {
                            "section": "body",
                        },
                    },
                ],
            )

            assert len(chunks) == 3

            await session.commit()

            stored_chunks = (
                await chunk_service.list_for_document(
                    document_id=document.id,
                )
            )

            assert len(stored_chunks) == 3

            assert [
                chunk.chunk_index
                for chunk in stored_chunks
            ] == [0, 1, 2]

            assert [
                chunk.text
                for chunk in stored_chunks
            ] == [
                "First chunk",
                "Second chunk",
                "Third chunk",
            ]

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_knowledge_chunk_single_retrieval() -> None:
    """Verify single chunk retrieval."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            document_service = KnowledgeDocumentService(
                session,
            )

            chunk_service = KnowledgeChunkService(
                session,
            )

            document = await document_service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="single-chunk-document",
            )

            chunk = await chunk_service.create_chunk(
                document_id=document.id,
                chunk_index=0,
                text="A single knowledge chunk.",
                content_hash="single-chunk-hash",
                token_count=5,
                metadata={
                    "page": 1,
                },
            )

            await session.commit()

            found = await chunk_service.get_by_id(
                chunk_id=chunk.id,
            )

            assert found is not None
            assert found.id == chunk.id
            assert found.document_id == document.id
            assert found.text == "A single knowledge chunk."
            assert found.content_hash == "single-chunk-hash"
            assert found.token_count == 5
            assert found.chunk_metadata["page"] == 1

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_knowledge_chunk_delete_for_document() -> None:
    """Verify deletion of all chunks belonging to a document."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            document_service = KnowledgeDocumentService(
                session,
            )

            chunk_service = KnowledgeChunkService(
                session,
            )

            document = await document_service.create_document(
                project_id=project_id,
                source_id=source_id,
                version=1,
                content_hash="delete-document",
            )

            await chunk_service.create_chunks(
                document_id=document.id,
                chunks=[
                    {
                        "chunk_index": 0,
                        "text": "Chunk A",
                        "content_hash": "delete-a",
                    },
                    {
                        "chunk_index": 1,
                        "text": "Chunk B",
                        "content_hash": "delete-b",
                    },
                ],
            )

            await session.commit()

            deleted_count = (
                await chunk_service.delete_for_document(
                    document_id=document.id,
                )
            )

            assert deleted_count == 2

            await session.commit()

            remaining = (
                await chunk_service.list_for_document(
                    document_id=document.id,
                )
            )

            assert remaining == []

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


# ============================================================
# Combined Persistence Tests
# ============================================================


@pytest.mark.asyncio
async def test_knowledge_persistence_service() -> None:
    """Verify atomic document + chunk persistence."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgePersistenceService(
                session,
            )

            document, chunks = (
                await service.persist_document_with_chunks(
                    project_id=project_id,
                    source_id=source_id,
                    version=1,
                    content_hash="combined-document-hash",
                    title="Combined Knowledge Document",
                    language="en",
                    status="COMPLETED",
                    metadata={
                        "pipeline": "knowledge",
                    },
                    chunks=[
                        {
                            "chunk_index": 0,
                            "text": "Introduction",
                            "content_hash": "combined-chunk-0",
                            "token_count": 1,
                            "metadata": {
                                "section": "introduction",
                            },
                        },
                        {
                            "chunk_index": 1,
                            "text": "Main content",
                            "content_hash": "combined-chunk-1",
                            "token_count": 2,
                            "metadata": {
                                "section": "body",
                            },
                        },
                    ],
                )
            )

            assert document.id is not None
            assert document.project_id == project_id
            assert document.source_id == source_id
            assert document.version == 1
            assert document.status == "COMPLETED"

            assert len(chunks) == 2

            assert all(
                chunk.document_id == document.id
                for chunk in chunks
            )

            await session.commit()

        async with SessionFactory() as session:
            stored_document = await session.get(
                KnowledgeDocument,
                document.id,
            )

            assert stored_document is not None

            statement = (
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.document_id
                    == document.id
                )
                .order_by(
                    KnowledgeChunk.chunk_index.asc()
                )
            )

            result = await session.execute(statement)

            stored_chunks = list(
                result.scalars().all()
            )

            assert len(stored_chunks) == 2
            assert stored_chunks[0].chunk_index == 0
            assert stored_chunks[1].chunk_index == 1

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_document_cascade_deletes_chunks() -> None:
    """Verify deleting a document removes its chunks."""

    user_id, project_id = await create_test_project()

    source_id = await create_test_source(
        project_id,
    )

    try:
        async with SessionFactory() as session:
            service = KnowledgePersistenceService(
                session,
            )

            document, chunks = (
                await service.persist_document_with_chunks(
                    project_id=project_id,
                    source_id=source_id,
                    version=1,
                    content_hash="cascade-document",
                    chunks=[
                        {
                            "chunk_index": 0,
                            "text": "Cascade test",
                            "content_hash": "cascade-chunk",
                        },
                    ],
                )
            )

            await session.commit()

            assert len(chunks) == 1

        async with SessionFactory() as session:
            document = await session.get(
                KnowledgeDocument,
                document.id,
            )

            assert document is not None

            await session.delete(document)
            await session.commit()

        async with SessionFactory() as session:
            statement = select(KnowledgeChunk).where(
                KnowledgeChunk.document_id
                == document.id
            )

            result = await session.execute(statement)

            remaining_chunks = list(
                result.scalars().all()
            )

            assert remaining_chunks == []

            deleted_document = await session.get(
                KnowledgeDocument,
                document.id,
            )

            assert deleted_document is None

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


# ============================================================
# Test Module Verification
# ============================================================


def test_knowledge_persistence_test_module() -> None:
    """Confirm the dedicated persistence test module is collected."""

    assert KnowledgeDocumentService is not None
    assert KnowledgeChunkService is not None
    assert KnowledgePersistenceService is not None

    print(
        "Knowledge persistence test module: READY"
    )