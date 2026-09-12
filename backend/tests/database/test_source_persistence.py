import asyncio
from uuid import uuid4

from app.database.session import SessionFactory
from app.ingestion.schemas import (
    IngestionRequest,
    InputType,
    ProcessingStatus,
)
from app.ingestion.source_service import SourcePersistenceService
from app.models.project import Project
from app.models.source import Source
from app.models.user import User


async def test_source_persistence() -> None:
    user_id = None
    project_id = None
    source_id = None

    # ---------------------------------------------------------
    # Create test user and project
    # ---------------------------------------------------------
    async with SessionFactory() as session:
        user = User(
            email=f"ingestion-test-{uuid4()}@example.com",
            name="Ingestion Test User",
        )

        session.add(user)
        await session.flush()

        user_id = user.id

        project = Project(
            owner_id=user.id,
            name="Ingestion Persistence Test",
            description="Temporary integration test project.",
            settings={},
        )

        session.add(project)
        await session.flush()

        project_id = project.id

        print("User creation: OK")
        print("Project creation: OK")

        # -----------------------------------------------------
        # Create ingestion request
        # -----------------------------------------------------
        request = IngestionRequest(
            project_id=project.id,
            source_id=uuid4(),
            input_type=InputType.TEXT,
            title="Test Source",
            filename="test.txt",
            mime_type="text/plain",
            content="Integration test content",
            metadata={
                "test": True,
                "environment": "integration",
            },
        )

        source_id = request.source_id

        # -----------------------------------------------------
        # Persist source
        # -----------------------------------------------------
        service = SourcePersistenceService(session)

        source = await service.create_source(
            request=request,
            storage_uri="file:///test/source.txt",
            content_hash="test-hash-123",
            status=ProcessingStatus.COMPLETED,
        )

        # -----------------------------------------------------
        # Validate generated source
        # -----------------------------------------------------
        assert source.id == source_id
        assert source.project_id == project.id
        assert source.source_type == InputType.TEXT.value
        assert source.title == "Test Source"
        assert source.original_filename == "test.txt"
        assert source.mime_type == "text/plain"
        assert source.storage_uri == "file:///test/source.txt"
        assert source.content_hash == "test-hash-123"

        assert (
            source.status
            == ProcessingStatus.COMPLETED.value
        )

        assert source.source_metadata["test"] is True
        assert (
            source.source_metadata["environment"]
            == "integration"
        )

        print("Source persistence: OK")
        print("Source identity consistency: OK")
        print("Metadata persistence: OK")
        print("Storage URI persistence: OK")
        print("Content hash persistence: OK")
        print("Status standardization: OK")

        # -----------------------------------------------------
        # Commit transaction
        # -----------------------------------------------------
        await session.commit()

    # ---------------------------------------------------------
    # Verify persistence using a fresh database session
    # ---------------------------------------------------------
    async with SessionFactory() as session:
        persisted_source = await session.get(
            Source,
            source_id,
        )

        assert persisted_source is not None

        assert persisted_source.id == source_id
        assert persisted_source.project_id == project_id
        assert persisted_source.source_type == InputType.TEXT.value
        assert persisted_source.title == "Test Source"
        assert persisted_source.original_filename == "test.txt"
        assert persisted_source.mime_type == "text/plain"
        assert (
            persisted_source.storage_uri
            == "file:///test/source.txt"
        )
        assert (
            persisted_source.content_hash
            == "test-hash-123"
        )

        assert (
            persisted_source.status
            == ProcessingStatus.COMPLETED.value
        )

        assert persisted_source.source_metadata["test"] is True
        assert (
            persisted_source.source_metadata["environment"]
            == "integration"
        )

        print("Source retrieval: OK")
        print("Fresh-session verification: OK")

        # -----------------------------------------------------
        # Cleanup source
        # -----------------------------------------------------
        await session.delete(persisted_source)

        # -----------------------------------------------------
        # Cleanup project
        # -----------------------------------------------------
        project_record = await session.get(
            Project,
            project_id,
        )

        if project_record is not None:
            await session.delete(project_record)

        # -----------------------------------------------------
        # Cleanup user
        # -----------------------------------------------------
        user_record = await session.get(
            User,
            user_id,
        )

        if user_record is not None:
            await session.delete(user_record)

        await session.commit()

    # ---------------------------------------------------------
    # Verify cleanup
    # ---------------------------------------------------------
    async with SessionFactory() as session:
        deleted_source = await session.get(
            Source,
            source_id,
        )

        deleted_project = await session.get(
            Project,
            project_id,
        )

        deleted_user = await session.get(
            User,
            user_id,
        )

        assert deleted_source is None
        assert deleted_project is None
        assert deleted_user is None

    print("Cleanup: OK")
    print("PostgreSQL source persistence: ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(test_source_persistence())