from __future__ import annotations

from uuid import UUID, uuid4

import pytest

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


async def create_test_project() -> tuple[UUID, UUID]:
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
    user_id: UUID,
    project_id: UUID,
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


async def get_source(
    source_id: UUID,
) -> Source | None:
    """Retrieve a source using a fresh session."""

    async with SessionFactory() as session:
        return await session.get(
            Source,
            source_id,
        )


def create_request(
    project_id: UUID,
    source_id: UUID | None = None,
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
# Test Doubles
# ============================================================


class FailingStorageService:
    """Storage service that fails during upload."""

    async def upload_source(self, **kwargs):
        raise RuntimeError(
            "Simulated storage upload failure"
        )


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
        self.compensated_storage_key = storage_key

        await super().compensate_upload(
            storage_key,
        )


class FailingPipeline:
    """Pipeline that always fails."""

    async def run(
        self,
        request: IngestionRequest,
    ):
        raise RuntimeError(
            "Simulated pipeline processing failure"
        )


class FailingCompensationService:
    """Compensation service that intentionally fails."""

    def __init__(self) -> None:
        self.storage_key: str | None = None

    async def compensate_upload(
        self,
        storage_key: str,
    ) -> None:
        self.storage_key = storage_key

        raise RuntimeError(
            "Simulated compensation failure"
        )


# ============================================================
# Test 1 — Storage Upload Failure
# ============================================================


@pytest.mark.asyncio
async def test_storage_upload_failure() -> None:
    """
    Verify that database persistence is not attempted
    when source storage fails.
    """

    user_id, project_id = (
        await create_test_project()
    )

    try:
        request = create_request(
            project_id=project_id,
        )

        service = IngestionApplicationService(
            session=None,
            storage=FailingStorageService(),
        )

        with pytest.raises(
            RuntimeError,
            match="Simulated storage upload failure",
        ):
            await service.ingest(
                request=request,
            )

        assert request.source_id is None

        # The application service should fail before
        # generating/persisting a source record.
        assert await get_source(
            request.source_id
        ) is None if request.source_id else True

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


# ============================================================
# Test 2 — Database Persistence Failure
# ============================================================


@pytest.mark.asyncio
async def test_database_persistence_failure() -> None:
    """
    Verify storage compensation after database failure.
    """

    storage = get_storage_service()

    user_id, project_id = (
        await create_test_project()
    )

    try:
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

            with pytest.raises(
                RuntimeError,
                match="Simulated database persistence failure",
            ):
                await service.ingest(
                    request=request,
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

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


# ============================================================
# Test 3 — Pipeline Processing Failure
# ============================================================


@pytest.mark.asyncio
async def test_pipeline_failure_source_retained() -> None:
    """
    Verify:

    - Source becomes FAILED.
    - Source remains in the database.
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

    storage_key = None

    try:
        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
            )

            service.pipeline = FailingPipeline()

            with pytest.raises(
                RuntimeError,
                match="Simulated pipeline processing failure",
            ):
                await service.ingest(
                    request=request,
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

        # ----------------------------------------------------
        # Verify original source remains in storage
        # ----------------------------------------------------

        storage_key = (
            f"source_file/"
            f"{source_id}/"
            f"{request.filename}"
        )

        assert await storage.exists(
            storage_key
        )

        stored_content = await storage.download(
            storage_key
        )

        assert (
            stored_content.decode("utf-8")
            == request.content
        )

    finally:
        # ----------------------------------------------------
        # Cleanup source file
        # ----------------------------------------------------

        if storage_key is not None:
            if await storage.exists(storage_key):
                await storage.delete(storage_key)

        # ----------------------------------------------------
        # Cleanup source record
        # ----------------------------------------------------

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


# ============================================================
# Test 4 — Compensation Failure
# ============================================================


@pytest.mark.asyncio
async def test_compensation_failure() -> None:
    """
    Verify that both the original database error and
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

    try:
        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
                compensation=compensation,
            )

            service.source_service = (
                FailingSourcePersistenceService()
            )

            with pytest.raises(
                StorageCompensationError
            ) as exc_info:
                await service.ingest(
                    request=request,
                )

        error = exc_info.value

        # ----------------------------------------------------
        # Storage key
        # ----------------------------------------------------

        assert error.storage_key is not None

        assert (
            error.storage_key
            == compensation.storage_key
        )

        # ----------------------------------------------------
        # Original database error
        # ----------------------------------------------------

        assert isinstance(
            error.original_error,
            RuntimeError,
        )

        assert (
            str(error.original_error)
            == "Simulated database persistence failure"
        )

        # ----------------------------------------------------
        # Compensation error
        # ----------------------------------------------------

        assert isinstance(
            error.compensation_error,
            RuntimeError,
        )

        assert (
            str(error.compensation_error)
            == "Simulated compensation failure"
        )

    finally:
        # ----------------------------------------------------
        # Compensation intentionally failed, so the uploaded
        # source file may still exist. Clean it up manually.
        # ----------------------------------------------------

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