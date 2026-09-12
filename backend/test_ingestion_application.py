from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from app.database.session import SessionFactory
from app.ingestion.application_service import IngestionApplicationService
from app.ingestion.schemas import IngestionRequest, InputType, ProcessingStatus
from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRResult,
    ASRStatus,
)
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.storage.dependencies import get_storage_service


# ============================================================
# Fake ASR Provider
# ============================================================


class FakeASRProvider:
    name = "fake_asr"

    async def transcribe(self, request: ASRRequest) -> ASRResult:
        return ASRResult(
            status=ASRStatus.COMPLETED,
            text="This is a fake transcript.",
            language="en",
            confidence=0.95,
            provider=self.name,
            metadata={
                "model": "fake-model",
            },
        )


# ============================================================
# Helpers
# ============================================================


async def create_test_user_and_project():
    """
    Create a temporary user and project for application-service tests.
    """

    user_id = uuid4()
    project_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"test-{user_id}@example.com",
            name="Test User",
        )

        project = Project(
            id=project_id,
            name=f"Test Project {project_id}",
            owner_id=user_id,
        )

        session.add(user)
        session.add(project)

        await session.commit()

    return user_id, project_id


async def cleanup_test_user_and_project(
    *,
    user_id,
    project_id,
):
    """
    Remove temporary user/project records.
    """

    async with SessionFactory() as session:
        project = await session.get(Project, project_id)
        user = await session.get(User, user_id)

        if project is not None:
            await session.delete(project)

        if user is not None:
            await session.delete(user)

        await session.commit()


# ============================================================
# Existing End-to-End Application Test
# ============================================================


@pytest.mark.asyncio
async def test_ingestion_application():
    """
    Verify the complete application-service ingestion lifecycle
    for a text source.

    Covers:

    Application Service
        ↓
    Storage
        ↓
    Source Persistence
        ↓
    Ingestion Pipeline
        ↓
    Canonical Content
        ↓
    Database
    """

    user_id = uuid4()
    project_id = uuid4()

    storage = get_storage_service()

    storage_key = None
    result = None

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"test-{user_id}@example.com",
            name="Test User",
        )

        project = Project(
            id=project_id,
            name=f"Test Project {project_id}",
            owner_id=user_id,
        )

        session.add(user)
        session.add(project)

        await session.commit()

        service = IngestionApplicationService(
            session=session,
            storage=storage,
        )

        request = IngestionRequest(
            project_id=project_id,
            input_type=InputType.TEXT,
            title="AI Communication",
            filename="article.txt",
            mime_type="text/plain",
            content=(
                "Artificial intelligence is transforming communication."
                "\n\n\n"
                "Organizations are using AI to create content faster."
            ),
        )

        result = await service.ingest(
    request=request,
)

        assert result.source_id is not None
        assert result.storage_key is not None
        assert result.storage_uri is not None
        assert result.content_hash is not None
        assert result.status == ProcessingStatus.COMPLETED

        storage_key = result.storage_key

        # ----------------------------------------------------
        # Verify canonical content
        # ----------------------------------------------------

        assert result.canonical_content is not None

        assert (
            result.canonical_content.text
            == "Artificial intelligence is transforming communication.\n\n"
            "Organizations are using AI to create content faster."
        )

        assert result.canonical_content.title == "AI Communication"

        assert result.canonical_content.source.source_id == result.source_id

        assert result.canonical_content.metadata is not None
        assert result.canonical_content.provenance is not None

    # --------------------------------------------------------
    # Verify database persistence using a fresh session
    # --------------------------------------------------------

    async with SessionFactory() as verify_session:
        source = await verify_session.get(Source, result.source_id)
        project = await verify_session.get(Project, project_id)
        user = await verify_session.get(User, user_id)

        assert source is not None
        assert project is not None
        assert user is not None

        assert source.project_id == project_id

    # --------------------------------------------------------
    # Verify stored content
    # --------------------------------------------------------

    stored_content = await storage.download(storage_key)

    assert (
    stored_content.decode("utf-8")
    == "Artificial intelligence is transforming communication."
    "\n\n\n"
    "Organizations are using AI to create content faster."
)

    # --------------------------------------------------------
    # Cleanup storage
    # --------------------------------------------------------

    await storage.delete(storage_key)

    # --------------------------------------------------------
    # Cleanup database
    # --------------------------------------------------------

    async with SessionFactory() as cleanup_session:
        source = await cleanup_session.get(Source, result.source_id)

        if source is not None:
            await cleanup_session.delete(source)

        project = await cleanup_session.get(Project, project_id)

        if project is not None:
            await cleanup_session.delete(project)

        user = await cleanup_session.get(User, user_id)

        if user is not None:
            await cleanup_session.delete(user)

        await cleanup_session.commit()

    # --------------------------------------------------------
    # Verify cleanup
    # --------------------------------------------------------

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


# ============================================================
# A3.5 — ASR Application Wiring
# ============================================================


@pytest.mark.asyncio
async def test_ingestion_application_injects_asr_provider():
    """
    Verify that an explicitly supplied ASR provider is propagated
    through the application service into the ingestion enrichment
    service.
    """

    user_id, project_id = await create_test_user_and_project()

    try:
        fake_asr_provider = FakeASRProvider()

        service = IngestionApplicationService(
            session=None,
            storage=get_storage_service(),
            asr_provider=fake_asr_provider,
        )

        # ----------------------------------------------------
        # Application service
        # ----------------------------------------------------

        assert service.asr_provider is fake_asr_provider

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        assert service.pipeline is not None

        # ----------------------------------------------------
        # Enrichment layer
        # ----------------------------------------------------

        assert service.pipeline.enrichment is not None

        assert (
            service.pipeline.enrichment.asr_provider
            is fake_asr_provider
        )

    finally:
        await cleanup_test_user_and_project(
            user_id=user_id,
            project_id=project_id,
        )


# ============================================================
# A3.5 — Explicit Provider Must Bypass Factory
# ============================================================


@pytest.mark.asyncio
async def test_ingestion_application_accepts_explicit_asr_provider():
    """
    Verify that an explicitly supplied ASR provider is used
    instead of creating another provider through the factory.
    """

    user_id, project_id = await create_test_user_and_project()

    try:
        fake_asr_provider = FakeASRProvider()

        with patch(
            "app.ingestion.application_service.create_asr_provider"
        ) as mock_create_asr_provider:

            service = IngestionApplicationService(
                session=None,
                storage=get_storage_service(),
                asr_provider=fake_asr_provider,
            )

            # Explicit dependency injection must win.
            assert service.asr_provider is fake_asr_provider

            assert (
                service.pipeline.enrichment.asr_provider
                is fake_asr_provider
            )

            # Factory must not be called.
            mock_create_asr_provider.assert_not_called()

    finally:
        await cleanup_test_user_and_project(
            user_id=user_id,
            project_id=project_id,
        )


# ============================================================
# A3.5 — Default Provider Factory
# ============================================================


@pytest.mark.asyncio
async def test_ingestion_application_creates_asr_provider_by_default():
    """
    Verify that the application service automatically creates
    the ASR provider through the configured ASR factory when
    no provider is explicitly supplied.
    """

    user_id, project_id = await create_test_user_and_project()

    try:
        fake_asr_provider = FakeASRProvider()

        with patch(
            "app.ingestion.application_service.create_asr_provider",
            return_value=fake_asr_provider,
        ) as mock_create_asr_provider:

            service = IngestionApplicationService(
                session=None,
                storage=get_storage_service(),
            )

            # ------------------------------------------------
            # Factory must be called exactly once.
            # ------------------------------------------------

            mock_create_asr_provider.assert_called_once()

            # ------------------------------------------------
            # Provider must be stored by application service.
            # ------------------------------------------------

            assert service.asr_provider is fake_asr_provider

            # ------------------------------------------------
            # Same provider must reach enrichment.
            # ------------------------------------------------

            assert service.pipeline.enrichment is not None

            assert (
                service.pipeline.enrichment.asr_provider
                is fake_asr_provider
            )

    finally:
        await cleanup_test_user_and_project(
            user_id=user_id,
            project_id=project_id,
        )