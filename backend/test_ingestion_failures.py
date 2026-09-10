import asyncio
from uuid import uuid4

from app.database.session import SessionFactory
from app.ingestion.application_service import (
    IngestionApplicationService,
)
from app.ingestion.schemas import (
    IngestionRequest,
    InputType,
    ProcessingStatus,
)
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.storage.compensation import (
    StorageCompensationService,
)
from app.storage.dependencies import (
    get_storage_service,
)
from app.storage.errors import (
    StorageCompensationError,
)


# ============================================================
# Test Helpers
# ============================================================


async def create_test_project() -> tuple:
    """Create a temporary user and project."""

    async with SessionFactory() as session:
        user = User(
            email=f"failure-test-{uuid4()}@example.com",
            name="Failure Test User",
            role="operator",
        )

        session.add(user)

        await session.flush()

        project = Project(
            owner_id=user.id,
            name="Failure Testing Project",
            description=(
                "Temporary project for failure-path testing"
            ),
        )

        session.add(project)

        await session.commit()

        return user.id, project.id


async def cleanup_test_project(
    user_id,
    project_id,
) -> None:
    """Remove temporary user and project."""

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


async def get_source(source_id):
    """Retrieve a source using a fresh session."""

    async with SessionFactory() as session:
        return await session.get(
            Source,
            source_id,
        )


def create_request(
    project_id,
    source_id=None,
) -> IngestionRequest:
    """Create a standard text ingestion request."""

    return IngestionRequest(
        project_id=project_id,
        source_id=source_id,
        input_type=InputType.TEXT,
        title="Failure Test Source",
        filename="failure-test.txt",
        mime_type="text/plain",
        content=(
            "This content is used to test "
            "ingestion failure handling."
        ),
        metadata={
            "test": True,
            "purpose": "failure-path",
        },
    )


# ============================================================
# Test 1 — Storage Upload Failure
# ============================================================


class FailingStorageService:
    """Storage service that fails during upload."""

    async def upload_source(self, **kwargs):
        raise RuntimeError(
            "Simulated storage upload failure"
        )


async def test_storage_upload_failure() -> None:
    """Verify DB is not touched when storage fails."""

    user_id, project_id = (
        await create_test_project()
    )

    request = create_request(
        project_id=project_id,
    )

    service = IngestionApplicationService(
        session=None,
        storage=FailingStorageService(),
    )

    try:
        await service.ingest(
            request=request,
        )

        raise AssertionError(
            "Expected storage failure "
            "was not raised."
        )

    except RuntimeError as exc:
        assert (
            str(exc)
            == "Simulated storage upload failure"
        )

    assert request.source_id is None

    await cleanup_test_project(
        user_id,
        project_id,
    )

    print(
        "Storage upload failure handling: OK"
    )


# ============================================================
# Test 2 — Database Persistence Failure
# ============================================================


class FailingSourcePersistenceService:
    """Persistence service that always fails."""

    async def create_source(self, **kwargs):
        raise RuntimeError(
            "Simulated database persistence failure"
        )


class TrackingCompensationService(
    StorageCompensationService
):
    """Track successful compensation."""

    def __init__(
        self,
        storage_service,
    ) -> None:
        super().__init__(
            storage_service,
        )

        self.compensation_called = False
        self.compensated_storage_key = None

    async def compensate_upload(
        self,
        storage_key: str,
    ) -> None:
        self.compensation_called = True

        self.compensated_storage_key = (
            storage_key
        )

        await super().compensate_upload(
            storage_key,
        )


async def test_database_persistence_failure() -> None:
    """
    Verify storage compensation after DB failure.
    """

    storage = get_storage_service()

    user_id, project_id = (
        await create_test_project()
    )

    request = create_request(
        project_id=project_id,
    )

    compensation = (
        TrackingCompensationService(
            storage_service=storage,
        )
    )

    async with SessionFactory() as session:
        service = IngestionApplicationService(
            session=session,
            storage=storage,
            compensation=compensation,
        )

        service.source_service = (
            FailingSourcePersistenceService()
        )

        try:
            await service.ingest(
                request=request,
            )

            raise AssertionError(
                "Expected database persistence "
                "failure was not raised."
            )

        except RuntimeError as exc:
            assert (
                str(exc)
                == "Simulated database "
                "persistence failure"
            )

    assert (
        compensation.compensation_called
        is True
    )

    assert (
        compensation.compensated_storage_key
        is not None
    )

    assert not await storage.exists(
        compensation.compensated_storage_key
    )

    await cleanup_test_project(
        user_id,
        project_id,
    )

    print(
        "Database persistence failure handling: OK"
    )

    print(
        "Storage compensation execution: OK"
    )

    print(
        "Orphan storage prevention: OK"
    )


# ============================================================
# Test 3 — Pipeline Processing Failure
# ============================================================


class FailingPipeline:
    """Pipeline that always fails."""

    async def run(self, request):
        raise RuntimeError(
            "Simulated pipeline processing failure"
        )


async def test_pipeline_failure_source_retained() -> None:
    """
    Verify:

    - Source becomes FAILED.
    - Source remains in database.
    - Original source file remains available.
    """

    storage = get_storage_service()

    user_id, project_id = (
        await create_test_project()
    )

    source_id = uuid4()

    request = create_request(
        project_id=project_id,
        source_id=source_id,
    )

    async with SessionFactory() as session:
        service = IngestionApplicationService(
            session=session,
            storage=storage,
        )

        service.pipeline = FailingPipeline()

        try:
            await service.ingest(
                request=request,
            )

            raise AssertionError(
                "Expected pipeline failure "
                "was not raised."
            )

        except RuntimeError as exc:
            assert (
                str(exc)
                == "Simulated pipeline "
                "processing failure"
            )

    source = await get_source(
        source_id,
    )

    assert source is not None

    assert (
        source.status
        == ProcessingStatus.FAILED.value
    )

    assert source.storage_uri is not None

    print(
        "Pipeline failure status FAILED: OK"
    )

    print(
        "Source database record retained: OK"
    )

    # --------------------------------------------------------
    # Verify original file
    # --------------------------------------------------------

    from pathlib import Path

    storage_root = Path(
        "./data/storage/source_file"
    )

    source_directory = (
        storage_root / str(source_id)
    )

    assert source_directory.exists()

    matching_files = [
        path
        for path in source_directory.iterdir()
        if path.is_file()
    ]

    assert len(matching_files) == 1

    retained_file = matching_files[0]

    assert retained_file.exists()

    assert (
        retained_file.read_text(
            encoding="utf-8"
        )
        == request.content
    )

    print(
        "Original source retention: OK"
    )

    # --------------------------------------------------------
    # Cleanup retained source
    # --------------------------------------------------------

    storage_key = (
        f"source_file/"
        f"{source_id}/"
        f"{retained_file.name}"
    )

    await storage.delete(
        storage_key,
    )

    assert not await storage.exists(
        storage_key,
    )

    print(
        "Failed-source storage cleanup: OK"
    )

    # --------------------------------------------------------
    # Cleanup DB
    # --------------------------------------------------------

    async with SessionFactory() as session:
        source = await session.get(
            Source,
            source_id,
        )

        if source is not None:
            await session.delete(source)

        await session.commit()

    await cleanup_test_project(
        user_id,
        project_id,
    )

    print(
        "Pipeline failure cleanup: OK"
    )


# ============================================================
# Test 4 — Compensation Failure
# ============================================================


class FailingCompensationService:
    """Compensation service that intentionally fails."""

    def __init__(self):
        self.storage_key = None

    async def compensate_upload(
        self,
        storage_key: str,
    ) -> None:
        self.storage_key = storage_key

        raise RuntimeError(
            "Simulated compensation failure"
        )


async def test_compensation_failure() -> None:
    """
    Verify that both the original database error and the
    compensation failure are preserved.
    """

    storage = get_storage_service()

    user_id, project_id = (
        await create_test_project()
    )

    request = create_request(
        project_id=project_id,
    )

    compensation = (
        FailingCompensationService()
    )

    async with SessionFactory() as session:
        service = IngestionApplicationService(
            session=session,
            storage=storage,
            compensation=compensation,
        )

        service.source_service = (
            FailingSourcePersistenceService()
        )

        try:
            await service.ingest(
                request=request,
            )

            raise AssertionError(
                "Expected compensation failure "
                "was not raised."
            )

        except StorageCompensationError as exc:

            # ------------------------------------------------
            # Verify storage key was preserved.
            # ------------------------------------------------

            assert exc.storage_key is not None

            assert (
                exc.storage_key
                == compensation.storage_key
            )

            # ------------------------------------------------
            # Verify original DB error.
            # ------------------------------------------------

            assert isinstance(
                exc.original_error,
                RuntimeError,
            )

            assert (
                str(exc.original_error)
                == "Simulated database "
                "persistence failure"
            )

            # ------------------------------------------------
            # Verify compensation error.
            # ------------------------------------------------

            assert isinstance(
                exc.compensation_error,
                RuntimeError,
            )

            assert (
                str(exc.compensation_error)
                == "Simulated compensation failure"
            )

    # --------------------------------------------------------
    # The failed compensation means the source file may still
    # exist. Clean it up manually for this test.
    # --------------------------------------------------------

    if compensation.storage_key is not None:
        if await storage.exists(
            compensation.storage_key
        ):
            await storage.delete(
                compensation.storage_key
            )

    await cleanup_test_project(
        user_id,
        project_id,
    )

    print(
        "Compensation failure detection: OK"
    )

    print(
        "Original database error preservation: OK"
    )

    print(
        "Compensation error preservation: OK"
    )

    print(
        "Compensation storage-key preservation: OK"
    )


# ============================================================
# Main Test Runner
# ============================================================


async def main() -> None:
    print()
    print(
        "=================================================="
    )
    print(
        "INGESTION FAILURE-PATH TESTS"
    )
    print(
        "=================================================="
    )
    print()

    await test_storage_upload_failure()

    await test_database_persistence_failure()

    await test_pipeline_failure_source_retained()

    await test_compensation_failure()

    print()
    print(
        "=================================================="
    )
    print(
        "FAILURE-PATH TESTS: ALL PASSED"
    )
    print(
        "=================================================="
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )