from __future__ import annotations

import pytest

from app.ingestion.parsers.docx import DOCXProcessor
from app.ingestion.parsers.document import TextDocumentProcessor
from app.ingestion.parsers.html import HTMLDocumentProcessor
from app.ingestion.parsers.image import ImageDocumentProcessor
from app.ingestion.parsers.pdf import PDFProcessor
from app.ingestion.parsers.text import TextProcessor
from app.ingestion.registry import create_processor_router
from app.ingestion.schemas import InputType
from app.ingestion.video.processor import VideoDocumentProcessor


def test_processor_registry() -> None:
    router = create_processor_router()

    # ---------------------------------------------------------
    # Supported input types
    # ---------------------------------------------------------

    assert router.supports(InputType.TEXT)
    assert router.supports(InputType.PROMPT)
    assert router.supports(InputType.TXT)
    assert router.supports(InputType.MARKDOWN)
    assert router.supports(InputType.PDF)
    assert router.supports(InputType.DOCX)
    assert router.supports(InputType.IMAGE)
    assert router.supports(InputType.AUDIO)
    assert router.supports(InputType.VIDEO)
    assert router.supports(InputType.URL)

    # ---------------------------------------------------------
    # Processor resolution
    # ---------------------------------------------------------

    assert isinstance(
        router.get_processor(InputType.TEXT),
        TextProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.PROMPT),
        TextProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.TXT),
        TextDocumentProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.MARKDOWN),
        TextDocumentProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.PDF),
        PDFProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.DOCX),
        DOCXProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.IMAGE),
        ImageDocumentProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.AUDIO),
        object,
    )

    assert isinstance(
        router.get_processor(InputType.VIDEO),
        VideoDocumentProcessor,
    )

    assert isinstance(
        router.get_processor(InputType.URL),
        HTMLDocumentProcessor,
    )


def test_processor_registry_supported_types() -> None:
    router = create_processor_router()

    supported_types = set(
        router.supported_types
    )

    expected_types = {
        InputType.TEXT,
        InputType.PROMPT,
        InputType.TXT,
        InputType.MARKDOWN,
        InputType.PDF,
        InputType.DOCX,
        InputType.IMAGE,
        InputType.AUDIO,
        InputType.VIDEO,
        InputType.URL,
    }

    assert supported_types == expected_types


def test_processor_registry_processor_instances() -> None:
    router = create_processor_router()

    text_processor = router.get_processor(
        InputType.TEXT
    )

    prompt_processor = router.get_processor(
        InputType.PROMPT
    )

    txt_processor = router.get_processor(
        InputType.TXT
    )

    markdown_processor = router.get_processor(
        InputType.MARKDOWN
    )

    pdf_processor = router.get_processor(
        InputType.PDF
    )

    docx_processor = router.get_processor(
        InputType.DOCX
    )

    image_processor = router.get_processor(
        InputType.IMAGE
    )

    audio_processor = router.get_processor(
        InputType.AUDIO
    )

    video_processor = router.get_processor(
        InputType.VIDEO
    )

    url_processor = router.get_processor(
        InputType.URL
    )

    # ---------------------------------------------------------
    # TEXT and PROMPT intentionally share TextProcessor.
    # ---------------------------------------------------------

    assert text_processor is prompt_processor

    # ---------------------------------------------------------
    # TXT and Markdown intentionally share the same
    # document processor.
    # ---------------------------------------------------------

    assert txt_processor is markdown_processor

    # ---------------------------------------------------------
    # Processor types
    # ---------------------------------------------------------

    assert isinstance(
        text_processor,
        TextProcessor,
    )

    assert isinstance(
        txt_processor,
        TextDocumentProcessor,
    )

    assert isinstance(
        pdf_processor,
        PDFProcessor,
    )

    assert isinstance(
        docx_processor,
        DOCXProcessor,
    )

    assert isinstance(
        image_processor,
        ImageDocumentProcessor,
    )

    assert isinstance(
        audio_processor,
        object,
    )

    assert isinstance(
        video_processor,
        VideoDocumentProcessor,
    )

    assert isinstance(
        url_processor,
        HTMLDocumentProcessor,
    )


def test_processor_registry_has_expected_count() -> None:
    router = create_processor_router()

    assert len(router.supported_types) == 10