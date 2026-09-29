from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.database.session import SessionFactory
from app.ingestion.application_service import IngestionApplicationService
from app.ingestion.fetchers.http import FetchedResponse
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
# Fake URL Fetcher
# ============================================================


@dataclass
class FakeURLFetcher:
    response: FetchedResponse

    async def fetch(self, url: str) -> FetchedResponse:
        return self.response


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
# URL Application-Level Ingestion
# ============================================================


@pytest.mark.asyncio
async def test_ingestion_application_fetches_url_before_processing():
    """
    Verify that URL ingestion:

    URL
        ↓
    HTTP Fetcher
        ↓
    Fetched HTML
        ↓
    Storage
        ↓
    Source Persistence
        ↓
    HTML Processor
        ↓
    Canonical Content
    """

    user_id = uuid4()
    project_id = uuid4()

    storage = get_storage_service()

    source_id = uuid4()
    storage_key = None
    result = None

    source_url = "https://example.com/article"
    final_url = "https://example.com/article/"
    html_content = (
        b"<html>"
        b"<head><title>AI Article</title></head>"
        b"<body>"
        b"<article>"
        b"<h1>Artificial Intelligence</h1>"
        b"<p>AI is transforming modern communication.</p>"
        b"<p>Organizations use AI to create content faster.</p>"
        b"</article>"
        b"</body>"
        b"</html>"
    )

    fake_fetcher = FakeURLFetcher(
        response=FetchedResponse(
            url=source_url,
            final_url=final_url,
            status_code=200,
            content_type="text/html",
            content=html_content,
            headers={
                "content-type": "text/html",
            },
        )
    )

    try:
        async with SessionFactory() as session:
            user = User(
                id=user_id,
                email=f"url-test-{user_id}@example.com",
                name="URL Test User",
            )

            project = Project(
                id=project_id,
                name=f"URL Test Project {project_id}",
                owner_id=user_id,
            )

            session.add(user)
            session.add(project)

            await session.commit()

            service = IngestionApplicationService(
                session=session,
                storage=storage,
                url_fetcher=fake_fetcher,
            )

            request = IngestionRequest(
                project_id=project_id,
                source_id=source_id,
                input_type=InputType.URL,
                title="AI Article",
                url=source_url,
                metadata={
                    "test_case": "url_application_ingestion",
                },
            )

            result = await service.ingest(
                request=request,
            )

            assert result.source_id == source_id
            assert result.status == ProcessingStatus.COMPLETED

            assert result.storage_key is not None
            assert result.storage_uri is not None
            assert result.content_hash is not None

            storage_key = result.storage_key

            # ------------------------------------------------
            # Canonical content must come from fetched HTML.
            # ------------------------------------------------

            assert result.canonical_content is not None

            canonical = result.canonical_content

            assert "AI is transforming modern communication." in canonical.text
            assert "Organizations use AI to create content faster." in (
                canonical.text
            )

            # ------------------------------------------------
            # Verify fetch metadata.
            # ------------------------------------------------

            assert canonical.metadata is not None

            assert canonical.metadata["source_url"] == source_url
            assert canonical.metadata["final_url"] == final_url
            assert canonical.metadata["http_status_code"] == 200
            assert canonical.metadata["fetched_content_type"] == "text/html"

            assert (
                canonical.metadata["test_case"]
                == "url_application_ingestion"
            )

        # ----------------------------------------------------
        # Verify database persistence.
        # ----------------------------------------------------

        async with SessionFactory() as verify_session:
            source = await verify_session.get(
                Source,
                source_id,
            )

            assert source is not None
            assert source.project_id == project_id
            assert source.storage_key == storage_key

        # ----------------------------------------------------
        # Verify the fetched bytes were stored.
        # ----------------------------------------------------

        assert storage_key is not None

        stored_content = await storage.download(storage_key)

        assert stored_content == html_content

    finally:
        # ----------------------------------------------------
        # Cleanup storage.
        # ----------------------------------------------------

        if storage_key is not None:
            await storage.delete(storage_key)

        # ----------------------------------------------------
        # Cleanup database.
        # ----------------------------------------------------

        async with SessionFactory() as cleanup_session:
            source = await cleanup_session.get(
                Source,
                source_id,
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


# ============================================================
# URL Fetch Failure
# ============================================================


@pytest.mark.asyncio
async def test_ingestion_application_rejects_url_when_fetch_fails():
    """
    Verify that a URL fetch failure is surfaced as a ValueError
    and does not silently continue into the ingestion pipeline.
    """

    user_id, project_id = await create_test_user_and_project()

    class FailingURLFetcher:
        async def fetch(self, url: str) -> FetchedResponse:
            raise RuntimeError("network failure")

    try:
        service = IngestionApplicationService(
            session=None,
            storage=get_storage_service(),
            url_fetcher=FailingURLFetcher(),
        )

        request = IngestionRequest(
            project_id=project_id,
            input_type=InputType.URL,
            title="Broken URL",
            url="https://example.com/broken",
        )

        with pytest.raises(
            ValueError,
            match="Failed to fetch URL: network failure",
        ):
            await service.ingest(
                request=request,
            )

    finally:
        await cleanup_test_user_and_project(
            user_id=user_id,
            project_id=project_id,
        )


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


# ============================================================
# A3.7.7 — Real ASR Application-Level E2E
# ============================================================


@pytest.mark.real_asr
@pytest.mark.asyncio
async def test_real_audio_ingestion_application_e2e():
    """
    Verify the complete production-style application flow for
    real audio ingestion with Faster-Whisper ASR.

    Covers:

    Real WAV
        ↓
    IngestionApplicationService
        ↓
    Storage
        ↓
    Source Persistence
        ↓
    Ingestion Pipeline
        ↓
    Audio Processor
        ↓
    FFprobe Media Inspection
        ↓
    Faster-Whisper ASR
        ↓
    Content Enrichment
        ↓
    Canonical Content
        ↓
    IngestionResult
        ↓
    Database Verification
    """

    from pathlib import Path

    from app.ingestion.speech.faster_whisper import (
        FasterWhisperASRProvider,
    )
    from app.ingestion.schemas import ContentBlockType

    user_id = uuid4()
    project_id = uuid4()

    storage = get_storage_service()

    audio_path = Path(r".\test_data\asr\sample_en.wav")
    audio = audio_path.read_bytes()

    storage_key = None
    result = None

    async with SessionFactory() as session:
        # ----------------------------------------------------
        # Create test user and project
        # ----------------------------------------------------

        user = User(
            id=user_id,
            email=f"real-asr-{user_id}@example.com",
            name="Real ASR Test User",
        )

        project = Project(
            id=project_id,
            name=f"Real ASR Test Project {project_id}",
            owner_id=user_id,
        )

        session.add(user)
        session.add(project)

        await session.commit()

        # ----------------------------------------------------
        # Use the real Faster-Whisper provider
        # ----------------------------------------------------

        asr_provider = FasterWhisperASRProvider(
            model_name="small",
            device="cpu",
            compute_type="int8",
        )

        service = IngestionApplicationService(
            session=session,
            storage=storage,
            asr_provider=asr_provider,
        )

        # ----------------------------------------------------
        # Real audio ingestion request
        # ----------------------------------------------------

        request = IngestionRequest(
            project_id=project_id,
            input_type=InputType.AUDIO,
            title="AI Communication Audio",
            filename="sample_en.wav",
            mime_type="audio/wav",
            content=audio,
            metadata={
                "asr_language": "en",
            },
        )

        # ----------------------------------------------------
        # Execute complete application flow
        # ----------------------------------------------------

        result = await service.ingest(
            request=request,
        )

        # ----------------------------------------------------
        # Verify ingestion result
        # ----------------------------------------------------

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

        canonical = result.canonical_content

        expected_text = (
            "Artificial intelligence is transforming communication. "
            "Organizations are using AI to create content faster."
        )

        assert canonical.text.strip() == expected_text

        assert canonical.title == "AI Communication Audio"

        assert canonical.language == "en"

        # ----------------------------------------------------
        # Verify source relationship
        # ----------------------------------------------------

        assert canonical.source.source_id == result.source_id

        # ----------------------------------------------------
        # Verify ASR metadata
        # ----------------------------------------------------

        assert canonical.metadata is not None
        assert canonical.metadata["asr"]["status"] == "completed"
        assert canonical.metadata["asr"]["provider"] == "faster_whisper"
        assert canonical.metadata["asr"]["segment_count"] == 2

        # ----------------------------------------------------
        # Verify transcript blocks
        # ----------------------------------------------------

        transcript_blocks = [
            block
            for block in canonical.segments
            if block.block_type == ContentBlockType.TRANSCRIPT
        ]

        assert len(transcript_blocks) == 2

        assert (
            transcript_blocks[0].content.strip()
            == "Artificial intelligence is transforming communication."
        )

        assert (
            transcript_blocks[1].content.strip()
            == "Organizations are using AI to create content faster."
        )

        # ----------------------------------------------------
        # Verify timestamps
        # ----------------------------------------------------

        for block in transcript_blocks:
            assert block.start_time is not None
            assert block.end_time is not None
            assert block.start_time < block.end_time

    # --------------------------------------------------------
    # Verify database persistence
    # --------------------------------------------------------

    async with SessionFactory() as verify_session:
        source = await verify_session.get(
            Source,
            result.source_id,
        )

        project = await verify_session.get(
            Project,
            project_id,
        )

        user = await verify_session.get(
            User,
            user_id,
        )

        assert source is not None
        assert project is not None
        assert user is not None

        assert source.project_id == project_id

    # --------------------------------------------------------
    # Verify stored original audio
    # --------------------------------------------------------

    assert storage_key is not None

    stored_audio = await storage.download(storage_key)

    assert stored_audio == audio

    # --------------------------------------------------------
    # Cleanup storage
    # --------------------------------------------------------

    await storage.delete(storage_key)

    # --------------------------------------------------------
    # Cleanup database
    # --------------------------------------------------------

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