from __future__ import annotations

from copy import deepcopy

from app.ingestion.schemas import (
    ContentBlockType,
    InputType,
    SourceReference,
)
from app.knowledge.normalization.schemas import (
    NormalizedKnowledgeDocument,
    NormalizedKnowledgeElement,
)
from app.knowledge.provenance import SourceMetadataBuilder


def make_source(
    source_type: InputType,
) -> SourceReference:
    return SourceReference(
        source_id="22222222-2222-2222-2222-222222222222",
        source_type=source_type,
        title="Test Source",
        filename="test.file",
        mime_type="application/octet-stream",
        content_hash="c" * 64,
        storage_uri="storage://test.file",
    )


def make_document(
    source_type: InputType,
    *,
    metadata: dict | None = None,
    elements: list[NormalizedKnowledgeElement] | None = None,
) -> NormalizedKnowledgeDocument:
    return NormalizedKnowledgeDocument(
        source=make_source(source_type),
        title="Test Document",
        language="en",
        text="Test normalized document.",
        elements=[] if elements is None else elements,
        entities=[],
        topics=[],
        claims=[],
        keywords=[],
        context={},
        provenance={},
        metadata={} if metadata is None else metadata,
        content_hash="a" * 64,
    )


def test_pdf_metadata_is_extracted() -> None:
    builder = SourceMetadataBuilder()

    document = make_document(
        InputType.PDF,
        metadata={
            "processor": "pdf",
            "page_count": 12,
            "extracted_page_count": 11,
            "unrelated": "value",
        },
    )

    result = builder.build(document=document)

    assert result.source_metadata.source_type == "pdf"

    assert result.source_metadata.metadata == {
        "builder": "source_metadata",
        "builder_version": "1.0",
        "processor": "pdf",
        "page_count": 12,
        "extracted_page_count": 11,
    }


def test_pdf_missing_metadata_is_safe() -> None:
    builder = SourceMetadataBuilder()

    document = make_document(InputType.PDF)

    result = builder.build(document=document)

    assert result.source_metadata.source_type == "pdf"

    assert result.source_metadata.metadata == {
        "builder": "source_metadata",
        "builder_version": "1.0",
    }


def test_docx_metadata_is_extracted_from_elements() -> None:
    builder = SourceMetadataBuilder()

    elements = [
        NormalizedKnowledgeElement(
            content="Chapter 1",
            block_type=ContentBlockType.HEADING,
            order=0,
            metadata={
                "style": "Heading 1",
                "heading_level": 1,
            },
        ),
        NormalizedKnowledgeElement(
            content="A paragraph.",
            block_type=ContentBlockType.PARAGRAPH,
            order=1,
            metadata={
                "style": "Normal",
            },
        ),
        NormalizedKnowledgeElement(
            content="A list item.",
            block_type=ContentBlockType.PARAGRAPH,
            order=2,
            metadata={
                "style": "List Paragraph",
                "list": True,
            },
        ),
    ]

    document = make_document(
        InputType.DOCX,
        elements=elements,
    )

    result = builder.build(document=document)

    metadata = result.source_metadata.metadata

    assert metadata["builder"] == "source_metadata"
    assert metadata["builder_version"] == "1.0"

    assert metadata["styles"] == [
        "Heading 1",
        "Normal",
        "List Paragraph",
    ]

    assert metadata["heading_levels"] == [1]
    assert metadata["list_block_count"] == 1


def test_docx_invalid_heading_level_is_ignored() -> None:
    builder = SourceMetadataBuilder()

    elements = [
        NormalizedKnowledgeElement(
            content="Invalid heading",
            block_type=ContentBlockType.HEADING,
            order=0,
            metadata={
                "heading_level": "invalid",
            },
        )
    ]

    document = make_document(
        InputType.DOCX,
        elements=elements,
    )

    result = builder.build(document=document)

    assert "heading_levels" not in (
        result.source_metadata.metadata
    )


def test_audio_metadata_is_extracted() -> None:
    builder = SourceMetadataBuilder()

    document = make_document(
        InputType.AUDIO,
        metadata={
            "media": {
                "duration": 180.5,
                "codec": "aac",
            },
            "duration": 180.5,
            "codec": "aac",
            "sample_rate": 44100,
            "channels": 2,
            "bitrate": 128000,
            "format": "mp3",
            "transcription_status": "completed",
            "unrelated": "ignored",
        },
    )

    result = builder.build(document=document)

    metadata = result.source_metadata.metadata

    assert metadata["media"] == {
        "duration": 180.5,
        "codec": "aac",
    }

    assert metadata["duration"] == 180.5
    assert metadata["codec"] == "aac"
    assert metadata["sample_rate"] == 44100
    assert metadata["channels"] == 2
    assert metadata["bitrate"] == 128000
    assert metadata["format"] == "mp3"
    assert metadata["transcription_status"] == "completed"


def test_video_metadata_is_extracted() -> None:
    builder = SourceMetadataBuilder()

    document = make_document(
        InputType.VIDEO,
        metadata={
            "media": {
                "duration": 600.0,
                "codec": "h264",
            },
            "duration": 600.0,
            "codec": "h264",
            "format": "mp4",
        },
    )

    result = builder.build(document=document)

    metadata = result.source_metadata.metadata

    assert metadata["media"]["duration"] == 600.0
    assert metadata["media"]["codec"] == "h264"

    assert metadata["duration"] == 600.0
    assert metadata["codec"] == "h264"
    assert metadata["format"] == "mp4"


def test_url_metadata_is_extracted() -> None:
    builder = SourceMetadataBuilder()

    document = make_document(
        InputType.URL,
        metadata={
            "source_url": "https://example.com/article",
            "canonical_url": "https://example.com/article",
            "title": "Example Article",
            "description": "Example description",
            "author": "Author",
            "language": "en",
            "keywords": ["ai", "rag"],
            "opengraph": {
                "site_name": "Example",
            },
            "article": {
                "published_time": "2026-01-01",
            },
            "unrelated": "ignored",
        },
    )

    result = builder.build(document=document)

    metadata = result.source_metadata.metadata

    assert metadata["source_url"] == (
        "https://example.com/article"
    )

    assert metadata["canonical_url"] == (
        "https://example.com/article"
    )

    assert metadata["title"] == "Example Article"

    assert metadata["description"] == (
        "Example description"
    )

    assert metadata["author"] == "Author"
    assert metadata["language"] == "en"
    assert metadata["keywords"] == ["ai", "rag"]

    assert metadata["opengraph"] == {
        "site_name": "Example",
    }

    assert metadata["article"] == {
        "published_time": "2026-01-01",
    }


def test_generic_source_preserves_metadata() -> None:
    builder = SourceMetadataBuilder()

    original = {
        "custom": "value",
        "nested": {
            "key": "value",
        },
    }

    document = make_document(
        InputType.TEXT,
        metadata=original,
    )

    result = builder.build(document=document)

    metadata = result.source_metadata.metadata

    assert metadata["custom"] == "value"
    assert metadata["nested"] == {
        "key": "value",
    }


def test_document_metadata_is_preserved_separately() -> None:
    builder = SourceMetadataBuilder()

    original = {
        "processor": "pdf",
        "page_count": 5,
        "custom": "value",
    }

    document = make_document(
        InputType.PDF,
        metadata=original,
    )

    result = builder.build(document=document)

    assert result.metadata == original


def test_builder_does_not_mutate_document_metadata() -> None:
    builder = SourceMetadataBuilder()

    original = {
        "media": {
            "duration": 100.0,
        },
        "custom": {
            "value": 123,
        },
    }

    document = make_document(
        InputType.VIDEO,
        metadata=original,
    )

    before = deepcopy(document.metadata)

    builder.build(document=document)

    assert document.metadata == before


def test_builder_result_is_deterministic() -> None:
    builder = SourceMetadataBuilder()

    document = make_document(
        InputType.PDF,
        metadata={
            "processor": "pdf",
            "page_count": 10,
        },
    )

    first = builder.build(document=document)
    second = builder.build(document=document)

    assert first == second
    assert first.model_dump() == second.model_dump()


def test_builder_does_not_share_nested_metadata() -> None:
    builder = SourceMetadataBuilder()

    original = {
        "media": {
            "duration": 120.0,
        },
    }

    document = make_document(
        InputType.VIDEO,
        metadata=original,
    )

    result = builder.build(document=document)

    assert result.metadata is not document.metadata

    assert (
        result.source_metadata.metadata["media"]
        is not document.metadata["media"]
    )