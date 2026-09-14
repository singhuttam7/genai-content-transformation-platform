from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.database.session import SessionFactory
from app.ingestion.application_service import (
    IngestionApplicationService,
)
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
    ProcessingStatus,
)
from app.ingestion.speech.schemas import (
    ASRResult,
    ASRSegment,
    ASRStatus,
)
from app.ingestion.video.schemas import (
    VideoASRResult,
    VideoASRStatus,
    VisionObservation,
    VisionResult,
    VisionStatus,
)
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.storage.dependencies import get_storage_service


# ============================================================
# Test fixture
# ============================================================

SAMPLE_VIDEO_PATH = Path(
    r".\test_data\video\sample_with_audio.mp4"
)


# ============================================================
# Fake Video ASR Service
# ============================================================


class FakeVideoASRService:
    """
    Deterministic VideoASRService test double.

    The application-level test verifies the complete
    ingestion architecture without invoking the real
    Faster-Whisper model.
    """

    def __init__(
        self,
        result: VideoASRResult,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def transcribe(
        self,
        media: bytes,
        *,
        video_info,
        request=None,
        filename: str | None = None,
    ) -> VideoASRResult:
        self.calls.append(
            {
                "media": media,
                "video_info": video_info,
                "request": request,
                "filename": filename,
            }
        )

        return self.result


# ============================================================
# Fake Video Vision Service
# ============================================================


class FakeVideoVisionService:
    """
    Deterministic VideoVisionService test double.

    The real Ollama/Gemma runtime is deliberately not used
    here because this test verifies application integration
    rather than local model performance.
    """

    def __init__(
        self,
        result: VisionResult,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def analyze(
        self,
        media: bytes,
        *,
        video_info,
        vision_request=None,
        frame_extraction_request=None,
    ) -> VisionResult:
        self.calls.append(
            {
                "media": media,
                "video_info": video_info,
                "vision_request": vision_request,
                "frame_extraction_request": (
                    frame_extraction_request
                ),
            }
        )

        return self.result


# ============================================================
# Result factories
# ============================================================


def make_completed_video_asr_result() -> VideoASRResult:
    """
    Create a deterministic successful Video ASR result.
    """

    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="This is a test video transcript.",
        segments=[
            ASRSegment(
                text="This is a test video transcript.",
                start_time=0.5,
                end_time=2.5,
                confidence=0.97,
            )
        ],
        language="en",
        confidence=0.97,
        provider="fake-video-asr",
        metadata={
            "model": "fake-model",
        },
    )

    return VideoASRResult(
        status=VideoASRStatus.COMPLETED,
        asr_result=asr_result,
        audio_extraction={
            "status": "completed",
            "provider": "fake-audio-extractor",
            "sample_rate": 16000,
            "channels": 1,
            "audio_format": "wav",
        },
        errors=[],
        metadata={
            "stage": "video_asr",
            "provider": "fake-video-asr",
        },
    )


def make_completed_vision_result() -> VisionResult:
    """
    Create a deterministic successful Vision result.
    """

    return VisionResult(
        status=VisionStatus.COMPLETED,
        observations=[
            VisionObservation(
                timestamp_seconds=1.0,
                frame_index=0,
                description=(
                    "A test scene is visible "
                    "in the video."
                ),
                objects=[
                    "scene",
                ],
                entities=[],
                actions=[],
                scene="test scene",
                visible_text=None,
                confidence=0.94,
                metadata={
                    "source": "fake-video-vision",
                },
            )
        ],
        requested_frames=1,
        processed_frames=1,
        failed_frames=0,
        errors=[],
        metadata={
            "provider": "fake-video-vision",
            "model": "fake-vision-model",
        },
    )


def make_failed_video_asr_result() -> VideoASRResult:
    """
    Create a deterministic failed Video ASR result.

    ASR capability failure must remain non-fatal to the
    overall video ingestion operation.
    """

    return VideoASRResult(
        status=VideoASRStatus.ASR_FAILED,
        asr_result=None,
        audio_extraction={
            "status": "completed",
            "provider": "fake-audio-extractor",
        },
        errors=[
            "Fake video ASR failure."
        ],
        metadata={
            "stage": "asr",
            "provider": "fake-video-asr",
            "error_type": "FakeASRFailure",
            "retryable": False,
        },
    )


def make_failed_vision_result() -> VisionResult:
    """
    Create a deterministic failed Vision result.

    Vision capability failure must remain non-fatal to the
    overall video ingestion operation.
    """

    return VisionResult(
        status=VisionStatus.FAILED,
        observations=[],
        requested_frames=1,
        processed_frames=0,
        failed_frames=1,
        errors=[
            "Fake video Vision failure."
        ],
        metadata={
            "service": "video_vision",
            "stage": "vision",
            "provider": "fake-video-vision",
            "error_type": "FakeVisionFailure",
            "retryable": False,
        },
    )


# ============================================================
# Database helpers
# ============================================================


async def create_test_user_and_project():
    """
    Create temporary database records required by the
    application-level ingestion test.
    """

    user_id = uuid4()
    project_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=(
                f"video-e2e-{user_id}@example.com"
            ),
            name="Video E2E Test User",
        )

        project = Project(
            id=project_id,
            name=(
                f"Video E2E Test Project "
                f"{project_id}"
            ),
            owner_id=user_id,
        )

        session.add(user)
        session.add(project)

        await session.commit()

    return user_id, project_id


async def cleanup_test_records(
    *,
    user_id,
    project_id,
    source_id,
):
    """
    Remove temporary database records created by the test.
    """

    async with SessionFactory() as session:
        source = await session.get(
            Source,
            source_id,
        )

        project = await session.get(
            Project,
            project_id,
        )

        user = await session.get(
            User,
            user_id,
        )

        if source is not None:
            await session.delete(source)

        if project is not None:
            await session.delete(project)

        if user is not None:
            await session.delete(user)

        await session.commit()


# ============================================================
# A4.8.2 — Complete Video Application Flow
# ============================================================


@pytest.mark.asyncio
async def test_video_ingestion_application_complete_flow():
    """
    Verify the complete application-level video ingestion flow.

    Coverage:

        Real MP4
            ↓
        IngestionApplicationService
            ↓
        Storage
            ↓
        Source Persistence
            ↓
        Ingestion Pipeline
            ↓
        Video Processor
            ↓
        FFprobe Metadata
            ↓
        Content Enrichment
            ├── Fake Video ASR
            └── Fake Video Vision
            ↓
        Canonical Content
            ↓
        Database Verification
            ↓
        Stored Source Verification
    """

    if not SAMPLE_VIDEO_PATH.exists():
        pytest.fail(
            "Required video fixture does not exist: "
            f"{SAMPLE_VIDEO_PATH}"
        )

    video_bytes = (
        SAMPLE_VIDEO_PATH.read_bytes()
    )

    user_id, project_id = (
        await create_test_user_and_project()
    )

    storage = get_storage_service()

    source_id = None
    storage_key = None

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    video_vision = FakeVideoVisionService(
        make_completed_vision_result()
    )

    request = IngestionRequest(
        project_id=project_id,
        input_type=InputType.VIDEO,
        title="Video Application E2E Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
        metadata={
            "asr_language": "en",
            "vision_detail_level": "standard",
            "vision_max_observations": 1,
            "test": True,
            "purpose": (
                "a4.8-application-video-integration"
            ),
        },
    )

    try:
        # =====================================================
        # Execute complete application flow
        # =====================================================

        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
                video_asr_service=video_asr,
                video_vision_service=video_vision,
            )

            result = await service.ingest(
                request=request,
            )

        # =====================================================
        # Basic ingestion result
        # =====================================================

        assert result is not None

        assert result.source_id is not None

        source_id = result.source_id

        assert result.storage_key is not None

        storage_key = result.storage_key

        assert result.storage_uri is not None
        assert result.content_hash is not None

        assert (
            result.status
            == ProcessingStatus.COMPLETED
        )

        # =====================================================
        # Verify canonical content
        # =====================================================

        canonical = (
            result.canonical_content
        )

        assert canonical is not None

        assert (
            canonical.title
            == "Video Application E2E Test"
        )

        assert (
            canonical.source.source_id
            == result.source_id
        )

        # =====================================================
        # Verify original VIDEO block
        # =====================================================

        video_blocks = [
            block
            for block in canonical.segments
            if block.block_type
            == ContentBlockType.VIDEO
        ]

        assert len(video_blocks) == 1

        # =====================================================
        # Verify transcript block
        # =====================================================

        transcript_blocks = [
            block
            for block in canonical.segments
            if block.block_type
            == ContentBlockType.TRANSCRIPT
        ]

        assert len(transcript_blocks) == 1

        transcript = (
            transcript_blocks[0]
        )

        assert (
            transcript.content
            == "This is a test video transcript."
        )

        assert (
            transcript.start_time
            == 0.5
        )

        assert (
            transcript.end_time
            == 2.5
        )

        assert (
            transcript.metadata["source"]
            == "asr"
        )

        # =====================================================
        # Verify combined canonical text
        # =====================================================

        assert (
            "This is a test video transcript."
            in canonical.text
        )

        # =====================================================
        # Verify Video ASR metadata
        # =====================================================

        assert (
            canonical.metadata[
                "video_asr"
            ]["status"]
            == "completed"
        )

        assert (
            canonical.metadata[
                "video_asr"
            ]["provider"]
            == "fake-video-asr"
        )

        # =====================================================
        # Verify Vision metadata
        # =====================================================

        assert (
            canonical.metadata[
                "video_vision"
            ]["status"]
            == "completed"
        )

        assert (
            canonical.metadata[
                "video_vision"
            ]["requested_frames"]
            == 1
        )

        assert (
            canonical.metadata[
                "video_vision"
            ]["processed_frames"]
            == 1
        )

        # =====================================================
        # Verify both video capabilities were invoked
        # =====================================================

        assert len(video_asr.calls) == 1
        assert len(video_vision.calls) == 1

        assert (
            video_asr.calls[0]["media"]
            == video_bytes
        )

        assert (
            video_asr.calls[0]["filename"]
            == "sample_with_audio.mp4"
        )

        assert (
            video_vision.calls[0]["media"]
            == video_bytes
        )

        assert (
            video_vision.calls[0]["video_info"]
            is not None
        )

        # =====================================================
        # Verify database persistence
        # =====================================================

        async with SessionFactory() as verify_session:
            source = await verify_session.get(
                Source,
                source_id,
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

            assert (
                source.project_id
                == project_id
            )

            assert (
                source.status
                == ProcessingStatus.COMPLETED.value
            )

            assert (
                source.content_hash
                == result.content_hash
            )

        # =====================================================
        # Verify original video in storage
        # =====================================================

        stored_video = (
            await storage.download(
                storage_key
            )
        )

        assert stored_video == video_bytes

    finally:
        # =====================================================
        # Storage cleanup
        # =====================================================

        if storage_key is not None:
            try:
                await storage.delete(
                    storage_key
                )
            except Exception:
                pass

        # =====================================================
        # Database cleanup
        # =====================================================

        if source_id is not None:
            await cleanup_test_records(
                user_id=user_id,
                project_id=project_id,
                source_id=source_id,
            )


# ============================================================
# A4.8.2 — Video ASR Failure
# ============================================================


@pytest.mark.asyncio
async def test_video_ingestion_application_survives_asr_failure():
    """
    Verify that a Video ASR capability failure remains
    non-fatal to the overall ingestion operation.

    Expected:

        Video processing     -> completed
        Video ASR            -> failed
        Vision               -> completed
        Canonical content    -> preserved
        Application result   -> completed
    """

    if not SAMPLE_VIDEO_PATH.exists():
        pytest.fail(
            "Required video fixture does not exist: "
            f"{SAMPLE_VIDEO_PATH}"
        )

    video_bytes = (
        SAMPLE_VIDEO_PATH.read_bytes()
    )

    user_id, project_id = (
        await create_test_user_and_project()
    )

    storage = get_storage_service()

    source_id = None
    storage_key = None

    video_asr = FakeVideoASRService(
        make_failed_video_asr_result()
    )

    video_vision = FakeVideoVisionService(
        make_completed_vision_result()
    )

    request = IngestionRequest(
        project_id=project_id,
        input_type=InputType.VIDEO,
        title="Video ASR Failure Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
    )

    try:
        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
                video_asr_service=video_asr,
                video_vision_service=video_vision,
            )

            result = await service.ingest(
                request=request,
            )

        source_id = result.source_id
        storage_key = result.storage_key

        assert (
            result.status
            == ProcessingStatus.COMPLETED
        )

        canonical = (
            result.canonical_content
        )

        assert canonical is not None

        # =====================================================
        # ASR failed
        # =====================================================

        assert (
            canonical.metadata[
                "video_asr"
            ]["status"]
            == "asr_failed"
        )

        # =====================================================
        # Vision still succeeded
        # =====================================================

        assert (
            canonical.metadata[
                "video_vision"
            ]["status"]
            == "completed"
        )

        # =====================================================
        # Verify Vision service was invoked
        # =====================================================

        assert len(video_vision.calls) == 1

        # =====================================================
        # Original video remains available
        # =====================================================

        assert any(
            block.block_type
            == ContentBlockType.VIDEO
            for block in canonical.segments
        )

    finally:
        if storage_key is not None:
            try:
                await storage.delete(
                    storage_key
                )
            except Exception:
                pass

        if source_id is not None:
            await cleanup_test_records(
                user_id=user_id,
                project_id=project_id,
                source_id=source_id,
            )


# ============================================================
# A4.8.2 — Video Vision Failure
# ============================================================


@pytest.mark.asyncio
async def test_video_ingestion_application_survives_vision_failure():
    """
    Verify that a Video Vision capability failure remains
    non-fatal to the overall ingestion operation.

    Expected:

        Video processing     -> completed
        Video ASR            -> completed
        Vision               -> failed
        Transcript           -> preserved
        Canonical content    -> preserved
        Application result   -> completed
    """

    if not SAMPLE_VIDEO_PATH.exists():
        pytest.fail(
            "Required video fixture does not exist: "
            f"{SAMPLE_VIDEO_PATH}"
        )

    video_bytes = (
        SAMPLE_VIDEO_PATH.read_bytes()
    )

    user_id, project_id = (
        await create_test_user_and_project()
    )

    storage = get_storage_service()

    source_id = None
    storage_key = None

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    video_vision = FakeVideoVisionService(
        make_failed_vision_result()
    )

    request = IngestionRequest(
        project_id=project_id,
        input_type=InputType.VIDEO,
        title="Video Vision Failure Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
    )

    try:
        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
                video_asr_service=video_asr,
                video_vision_service=video_vision,
            )

            result = await service.ingest(
                request=request,
            )

        source_id = result.source_id
        storage_key = result.storage_key

        assert (
            result.status
            == ProcessingStatus.COMPLETED
        )

        canonical = (
            result.canonical_content
        )

        assert canonical is not None

        # =====================================================
        # Vision failed
        # =====================================================

        assert (
            canonical.metadata[
                "video_vision"
            ]["status"]
            == "failed"
        )

        # =====================================================
        # ASR still succeeded
        # =====================================================

        assert (
            canonical.metadata[
                "video_asr"
            ]["status"]
            == "completed"
        )

        # =====================================================
        # Verify Vision service was invoked
        # =====================================================

        assert len(video_vision.calls) == 1

        # =====================================================
        # Verify transcript remains available
        # =====================================================

        transcript_blocks = [
            block
            for block in canonical.segments
            if block.block_type
            == ContentBlockType.TRANSCRIPT
        ]

        assert len(transcript_blocks) == 1

        assert (
            transcript_blocks[0].content
            == "This is a test video transcript."
        )

        # =====================================================
        # Original video remains available
        # =====================================================

        assert any(
            block.block_type
            == ContentBlockType.VIDEO
            for block in canonical.segments
        )

    finally:
        if storage_key is not None:
            try:
                await storage.delete(
                    storage_key
                )
            except Exception:
                pass

        if source_id is not None:
            await cleanup_test_records(
                user_id=user_id,
                project_id=project_id,
                source_id=source_id,
            )