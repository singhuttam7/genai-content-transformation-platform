import asyncio
from uuid import uuid4

from app.database.session import SessionFactory
from app.ingestion.application_service import IngestionApplicationService
from app.ingestion.schemas import IngestionRequest, InputType
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.storage.dependencies import get_storage_service


async def test_ingestion_application() -> None:
    storage = get_storage_service()

    user_id = None
    project_id = None
    result = None
    storage_key = None

    # =========================================================
    # 1. Create temporary user and project
    # =========================================================
    async with SessionFactory() as session:
        user = User(
            email=f"ingestion-test-{uuid4()}@example.com",
            name="Ingestion Test User",
            role="operator",
        )

        session.add(user)
        await session.flush()

        project = Project(
            owner_id=user.id,
            name="Ingestion Integration Test",
            description="Temporary project for ingestion testing",
        )

        session.add(project)
        await session.commit()

        user_id = user.id
        project_id = project.id

    print("User creation: OK")
    print("Project creation: OK")

    # =========================================================
    # 2. Prepare ingestion request
    # =========================================================
    content = (
        "Artificial intelligence is transforming communication.\n\n\n"
        "Organizations are using AI to create content faster."
    )

    request = IngestionRequest(
        project_id=project_id,
        input_type=InputType.TEXT,
        title="AI Communication",
        filename="integration-test.txt",
        mime_type="text/plain",
        content=content,
        metadata={
            "test": True,
            "source": "integration-test",
        },
    )

    # =========================================================
    # 3. Run unified ingestion application service
    # =========================================================
    async with SessionFactory() as session:
        service = IngestionApplicationService(
            session=session,
            storage=storage,
        )

        print("Application service creation: OK")

        result = await service.ingest(request=request)

        print("Source storage: OK")
        print("Source database persistence: OK")
        print("Content processing: OK")
        print("Canonical content generation: OK")
        print("Source hash generation: OK")

    # =========================================================
    # 4. Validate ingestion result
    # =========================================================
    assert result is not None
    assert result.source_id is not None
    assert result.storage_key is not None
    assert result.storage_uri is not None
    assert result.content_hash is not None
    assert result.status == "COMPLETED"

    storage_key = result.storage_key

    expected_text = (
        "Artificial intelligence is transforming communication.\n\n"
        "Organizations are using AI to create content faster."
    )

    assert result.canonical_content.text == expected_text

    assert result.canonical_content.title == "AI Communication"

    assert result.canonical_content.source.source_id == result.source_id

    assert result.canonical_content.metadata["test"] is True
    assert (
        result.canonical_content.metadata["source"]
        == "integration-test"
    )

    assert (
        result.canonical_content.provenance["ingestion_normalizer"]
        == "default"
    )

    assert (
        result.canonical_content.provenance["normalization_version"]
        == "1.0"
    )

    print("Ingestion result validation: OK")

    # =========================================================
    # 5. Verify database persistence using a fresh session
    # =========================================================
    async with SessionFactory() as verify_session:
        source = await verify_session.get(
            Source,
            result.source_id,
        )

        assert source is not None
        assert source.id == result.source_id
        assert source.project_id == project_id

        assert source.source_type == InputType.TEXT.value
        assert source.title == "AI Communication"
        assert source.original_filename == "integration-test.txt"
        assert source.mime_type == "text/plain"

        assert source.status == "COMPLETED"

        assert source.content_hash == result.content_hash
        assert source.storage_uri == result.storage_uri

        assert source.source_metadata["test"] is True
        assert (
            source.source_metadata["source"]
            == "integration-test"
        )

        print("Source status COMPLETED: OK")
        print("Database verification: OK")

    # =========================================================
    # 6. Verify stored source file
    # =========================================================
    assert storage_key is not None

    exists = await storage.exists(storage_key)

    assert exists is True

    stored_content = await storage.download(storage_key)

    assert stored_content.decode("utf-8") == content

    print("Stored source verification: OK")

    # =========================================================
    # 7. Cleanup storage
    # =========================================================
    await storage.delete(storage_key)

    assert not await storage.exists(storage_key)

    print("Storage cleanup: OK")

    # =========================================================
    # 8. Cleanup database
    # =========================================================
    async with SessionFactory() as cleanup_session:
        source = await cleanup_session.get(
            Source,
            result.source_id,
        )

        project = await cleanup_session.get(
            Project,
            project_id,
        )

        user = await cleanup_session.get(
            User,
            user_id,
        )

        if source is not None:
            await cleanup_session.delete(source)

        if project is not None:
            await cleanup_session.delete(project)

        if user is not None:
            await cleanup_session.delete(user)

        await cleanup_session.commit()

    print("Database cleanup: OK")

    # =========================================================
    # 9. Verify database cleanup
    # =========================================================
    async with SessionFactory() as verify_cleanup_session:
        source = await verify_cleanup_session.get(
            Source,
            result.source_id,
        )

        project = await verify_cleanup_session.get(
            Project,
            project_id,
        )

        user = await verify_cleanup_session.get(
            User,
            user_id,
        )

        assert source is None
        assert project is None
        assert user is None

    print("Cleanup verification: OK")

    # =========================================================
    # Final result
    # =========================================================
    print()
    print("Unified ingestion application: ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(test_ingestion_application())