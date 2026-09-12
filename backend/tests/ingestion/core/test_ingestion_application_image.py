from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw, ImageFont

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
from app.storage.dependencies import get_storage_service


def create_test_image() -> bytes:
    """
    Create a deterministic PNG image containing text
    for real Tesseract OCR integration testing.
    """

    image = Image.new(
        "RGB",
        (1000, 300),
        "white",
    )

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            52,
        )
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (60, 70),
        "GEN AI PLATFORM",
        fill="black",
        font=font,
    )

    draw.text(
        (60, 160),
        "OCR INTEGRATION TEST",
        fill="black",
        font=font,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


@pytest.mark.asyncio
async def test_real_image_ingestion_with_tesseract() -> None:
    """
    Verify the complete production image-ingestion path.

    Flow:

        Image bytes
            ↓
        Application Service
            ↓
        Storage Upload
            ↓
        PostgreSQL Source
            ↓
        Ingestion Pipeline
            ↓
        Image Processor
            ↓
        Storage Content Resolver
            ↓
        Tesseract OCR
            ↓
        Content Enrichment
            ↓
        Normalizer
            ↓
        CanonicalContent
            ↓
        Source COMPLETED
    """

    storage = get_storage_service()

    user_id = None
    project_id = None
    source_id = None
    storage_key = None

    image_bytes = create_test_image()

    # =========================================================
    # 1. Create temporary user and project
    # =========================================================

    async with SessionFactory() as session:
        user = User(
            email=f"image-ingestion-{uuid4()}@example.com",
            name="Image Ingestion Test User",
            role="operator",
        )

        session.add(user)

        await session.flush()

        project = Project(
            owner_id=user.id,
            name="Image OCR Integration Test",
            description=(
                "Temporary project for real Tesseract "
                "image ingestion testing"
            ),
        )

        session.add(project)

        await session.commit()

        user_id = user.id
        project_id = project.id

    # =========================================================
    # 2. Prepare image ingestion request
    # =========================================================

    request = IngestionRequest(
        project_id=project_id,
        input_type=InputType.IMAGE,
        title="OCR Integration Image",
        filename="ocr-integration-test.png",
        mime_type="image/png",
        content=image_bytes,
        metadata={
            "test": True,
            "purpose": "real-tesseract-integration",
        },
    )

    # =========================================================
    # 3. Execute real application service
    # =========================================================

    try:
        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
            )

            result = await service.ingest(
                request=request,
            )

        # =====================================================
        # 4. Basic ingestion result
        # =====================================================

        assert result is not None

        assert result.source_id is not None

        assert result.storage_key is not None
        assert result.storage_uri is not None
        assert result.content_hash is not None

        assert (
            result.status
            == ProcessingStatus.COMPLETED
        )

        source_id = result.source_id
        storage_key = result.storage_key

        # =====================================================
        # 5. Verify OCR text
        # =====================================================

        canonical_text = (
            result.canonical_content.text.upper()
        )

        assert (
    "GEN AI PLATFORM" in canonical_text
    or "GEN AL PLATFORM" in canonical_text
)

        assert "OCR INTEGRATION TEST" in canonical_text

        # =====================================================
        # 6. Verify canonical source information
        # =====================================================

        canonical_source = (
            result.canonical_content.source
        )

        assert (
            canonical_source.source_id
            == result.source_id
        )

        assert (
            canonical_source.source_type
            == InputType.IMAGE
        )

        assert (
            canonical_source.filename
            == "ocr-integration-test.png"
        )

        assert (
            canonical_source.mime_type
            == "image/png"
        )

        # =====================================================
        # 7. Verify OCR metadata
        # =====================================================

        ocr_metadata = (
            result.canonical_content.metadata[
                "ocr"
            ]
        )

        assert (
            ocr_metadata["status"]
            == "completed"
        )

        assert (
            ocr_metadata["provider"]
            == "tesseract"
        )

        assert (
            ocr_metadata["language"]
            == "eng"
        )

        assert (
            ocr_metadata["block_count"]
            > 0
        )

        assert (
            ocr_metadata["confidence"]
            is not None
        )

        assert (
            ocr_metadata["confidence"]
            >= 0
        )

        # =====================================================
        # 8. Verify OCR blocks
        # =====================================================

        ocr_blocks = [
            block
            for block in result.canonical_content.segments
            if block.metadata.get("source")
            == "ocr"
        ]

        assert len(ocr_blocks) > 0

        for block in ocr_blocks:
            assert block.content.strip()

            assert (
                block.metadata["provider"]
                == "tesseract"
            )

            assert (
                block.metadata["bounding_box"]
                is not None
            )

            assert (
                block.metadata["confidence"]
                is not None
            )

        # =====================================================
        # 9. Verify original image block is preserved
        # =====================================================

        image_blocks = [
            block
            for block in result.canonical_content.segments
            if block.block_type.value == "image"
        ]

        assert len(image_blocks) == 1

        # =====================================================
        # 10. Verify database persistence
        # =====================================================

        async with SessionFactory() as verify_session:
            source = await verify_session.get(
                Source,
                result.source_id,
            )

            assert source is not None

            assert source.id == result.source_id

            assert (
                source.project_id
                == project_id
            )

            assert (
                source.source_type
                == InputType.IMAGE.value
            )

            assert (
                source.title
                == "OCR Integration Image"
            )

            assert (
                source.original_filename
                == "ocr-integration-test.png"
            )

            assert (
                source.mime_type
                == "image/png"
            )

            assert (
                source.status
                == ProcessingStatus.COMPLETED.value
            )

            assert (
                source.content_hash
                == result.content_hash
            )

            assert (
                source.storage_uri
                == result.storage_uri
            )

            assert (
                source.source_metadata["test"]
                is True
            )

            assert (
                source.source_metadata["purpose"]
                == "real-tesseract-integration"
            )

        # =====================================================
        # 11. Verify original image exists in storage
        # =====================================================

        assert storage_key is not None

        assert await storage.exists(
            storage_key
        )

        stored_image = await storage.download(
            storage_key
        )

        assert stored_image == image_bytes

    finally:

        # =====================================================
        # 12. Cleanup storage
        # =====================================================

        if storage_key is not None:

            if await storage.exists(
                storage_key
            ):
                await storage.delete(
                    storage_key
                )

        # =====================================================
        # 13. Cleanup database source
        # =====================================================

        if source_id is not None:

            async with SessionFactory() as session:

                source = await session.get(
                    Source,
                    source_id,
                )

                if source is not None:
                    await session.delete(
                        source
                    )

                await session.commit()

        # =====================================================
        # 14. Cleanup project and user
        # =====================================================

        if project_id is not None:

            async with SessionFactory() as session:

                project = await session.get(
                    Project,
                    project_id,
                )

                user = None

                if user_id is not None:
                    user = await session.get(
                        User,
                        user_id,
                    )

                if project is not None:
                    await session.delete(
                        project
                    )

                if user is not None:
                    await session.delete(
                        user
                    )

                await session.commit()

        # =====================================================
        # 15. Verify cleanup
        # =====================================================

        if source_id is not None:

            async with SessionFactory() as session:

                source = await session.get(
                    Source,
                    source_id,
                )

                assert source is None

        if project_id is not None:

            async with SessionFactory() as session:

                project = await session.get(
                    Project,
                    project_id,
                )

                assert project is None

        if user_id is not None:

            async with SessionFactory() as session:

                user = await session.get(
                    User,
                    user_id,
                )

                assert user is None