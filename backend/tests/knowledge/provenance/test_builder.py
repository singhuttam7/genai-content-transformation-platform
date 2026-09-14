from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.ingestion.schemas import (
    ContentBlockType,
    InputType,
    SourceReference,
)
from app.knowledge.chunking.schemas import (
    ChunkSourceReference,
    KnowledgeChunkDraft,
)
from app.knowledge.normalization.schemas import (
    NormalizedKnowledgeDocument,
)
from app.knowledge.provenance import (
    KnowledgeChunkMetadata,
    KnowledgeProvenanceBuilder,
)


PROJECT_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

SOURCE_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)

DOCUMENT_ID = UUID(
    "33333333-3333-3333-3333-333333333333"
)

DOCUMENT_HASH = "a" * 64
CHUNK_HASH = "b" * 64


def make_document() -> NormalizedKnowledgeDocument:
    """Create a deterministic normalized knowledge document."""

    source = SourceReference(
        source_id=SOURCE_ID,
        source_type=InputType.PDF,
        title="Research Report",
        filename="report.pdf",
        mime_type="application/pdf",
        content_hash="c" * 64,
        storage_uri="storage://documents/report.pdf",
    )

    return NormalizedKnowledgeDocument(
        source=source,
        title="Research Report",
        language="en",
        text=(
            "This is the normalized document text. "
            "It contains multiple sections."
        ),
        elements=[],
        entities=["Entity A", "Entity B"],
        topics=["AI", "Research"],
        claims=["Claim A"],
        keywords=[
            "artificial intelligence",
            "research",
        ],
        context={
            "domain": "research",
        },
        provenance={
            "source_system": "ingestion",
        },
        metadata={
            "author": "Test Author",
        },
        content_hash=DOCUMENT_HASH,
    )


def make_chunk(
    *,
    metadata: dict | None = None,
    page_numbers: list[int] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> KnowledgeChunkDraft:
    """Create a deterministic knowledge chunk draft."""

    source = ChunkSourceReference(
        element_orders=[3, 4],
        block_types=[
            ContentBlockType.HEADING,
            ContentBlockType.PARAGRAPH,
        ],
        page_numbers=(
            [4]
            if page_numbers is None
            else page_numbers
        ),
        start_time=start_time,
        end_time=end_time,
        section_path=[
            "Chapter 2",
            "Methodology",
        ],
    )

    return KnowledgeChunkDraft(
        chunk_index=7,
        text="This is a knowledge chunk.",
        content_hash=CHUNK_HASH,
        token_count=6,
        source=source,
        metadata=(
            {
                "chunking_strategy": "structure_aware",
                "chunking_strategy_version": "1.0",
                "custom_field": "preserved",
            }
            if metadata is None
            else metadata
        ),
    )


def test_build_returns_complete_metadata() -> None:
    builder = KnowledgeProvenanceBuilder()

    result = builder.build(
        document=make_document(),
        chunk=make_chunk(),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=3,
    )

    assert isinstance(
        result,
        KnowledgeChunkMetadata,
    )

    assert result.chunk_index == 7

    assert result.character_count == len(
        "This is a knowledge chunk."
    )

    assert result.token_count == 6
    assert result.content_hash == CHUNK_HASH

    assert (
        result.chunking_strategy
        == "structure_aware"
    )

    assert (
        result.chunking_strategy_version
        == "1.0"
    )


def test_source_provenance_is_correct() -> None:
    builder = KnowledgeProvenanceBuilder()
    document = make_document()

    result = builder.build(
        document=document,
        chunk=make_chunk(),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    source = result.provenance.source

    assert source.project_id == PROJECT_ID
    assert source.source_id == SOURCE_ID
    assert source.source_type == "pdf"

    assert (
        source.source_uri
        == "storage://documents/report.pdf"
    )

    assert source.title == "Research Report"
    assert source.language == "en"


def test_document_provenance_uses_document_hash() -> None:
    builder = KnowledgeProvenanceBuilder()
    document = make_document()

    result = builder.build(
        document=document,
        chunk=make_chunk(),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=4,
    )

    provenance = result.provenance.document

    assert provenance.document_id == DOCUMENT_ID
    assert provenance.document_version == 4

    # Document provenance must contain the
    # normalized document hash.
    assert provenance.content_hash == DOCUMENT_HASH

    # Chunk metadata must contain the chunk hash.
    assert result.content_hash == CHUNK_HASH

    # These hashes must remain distinct.
    assert (
        provenance.content_hash
        != result.content_hash
    )


def test_location_provenance_preserves_structure() -> None:
    builder = KnowledgeProvenanceBuilder()

    result = builder.build(
        document=make_document(),
        chunk=make_chunk(),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    location = result.provenance.location

    assert location.element_orders == [3, 4]

    assert location.block_types == [
        ContentBlockType.HEADING,
        ContentBlockType.PARAGRAPH,
    ]

    assert location.page_numbers == [4]

    assert location.section_path == [
        "Chapter 2",
        "Methodology",
    ]


def test_timestamp_provenance_is_preserved() -> None:
    builder = KnowledgeProvenanceBuilder()

    result = builder.build(
        document=make_document(),
        chunk=make_chunk(
            start_time=42.5,
            end_time=58.75,
        ),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    location = result.provenance.location

    assert location.start_time == 42.5
    assert location.end_time == 58.75


def test_metadata_is_preserved_with_source_metadata() -> None:
    builder = KnowledgeProvenanceBuilder()

    original_metadata = {
        "chunking_strategy": "structure_aware",
        "chunking_strategy_version": "1.0",
        "custom_field": "preserved",
    }

    document = make_document()

    result = builder.build(
        document=document,
        chunk=make_chunk(
            metadata=original_metadata
        ),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    # Existing chunk metadata remains intact.
    assert (
        result.metadata["chunking_strategy"]
        == "structure_aware"
    )

    assert (
        result.metadata[
            "chunking_strategy_version"
        ]
        == "1.0"
    )

    assert (
        result.metadata["custom_field"]
        == "preserved"
    )

    # Source-specific metadata is now added.
    assert "source_metadata" in result.metadata

    source_metadata = result.metadata[
        "source_metadata"
    ]

    assert source_metadata["source_type"] == "pdf"

    assert (
        source_metadata["metadata"]["builder"]
        == "source_metadata"
    )

    assert (
        source_metadata["metadata"][
            "builder_version"
        ]
        == "1.0"
    )


def test_metadata_is_copied_not_shared() -> None:
    builder = KnowledgeProvenanceBuilder()

    original_metadata = {
        "chunking_strategy": "structure_aware",
        "chunking_strategy_version": "1.0",
        "custom": {
            "value": 123,
        },
    }

    chunk = make_chunk(
        metadata=original_metadata,
    )

    result = builder.build(
        document=make_document(),
        chunk=chunk,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    # Original chunk metadata must remain untouched.
    assert chunk.metadata == original_metadata

    # Result receives a separate metadata dictionary.
    assert result.metadata is not chunk.metadata

    # Existing nested metadata remains intact.
    assert result.metadata["custom"] == {
        "value": 123,
    }

    # Source metadata is added independently.
    assert "source_metadata" in result.metadata

    assert (
        result.metadata["source_metadata"][
            "source_type"
        ]
        == "pdf"
    )


def test_metadata_source_metadata_is_not_same_as_document_metadata() -> None:
    builder = KnowledgeProvenanceBuilder()

    document = make_document()

    result = builder.build(
        document=document,
        chunk=make_chunk(),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    source_metadata = result.metadata[
        "source_metadata"
    ]

    assert (
        source_metadata
        is not document.metadata
    )


def test_build_is_deterministic() -> None:
    builder = KnowledgeProvenanceBuilder()

    document = make_document()
    chunk = make_chunk()

    first = builder.build(
        document=document,
        chunk=chunk,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=2,
    )

    second = builder.build(
        document=document,
        chunk=chunk,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=2,
    )

    assert first == second

    assert (
        first.model_dump()
        == second.model_dump()
    )


def test_builder_does_not_mutate_inputs() -> None:
    builder = KnowledgeProvenanceBuilder()

    document = make_document()
    chunk = make_chunk()

    document_before = document.model_dump(
        mode="json"
    )

    chunk_before = chunk.model_dump(
        mode="json"
    )

    builder.build(
        document=document,
        chunk=chunk,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    assert (
        document.model_dump(mode="json")
        == document_before
    )

    assert (
        chunk.model_dump(mode="json")
        == chunk_before
    )


def test_missing_chunking_metadata_uses_unknown() -> None:
    builder = KnowledgeProvenanceBuilder()

    chunk = make_chunk(
        metadata={
            "custom_field": "value",
        }
    )

    result = builder.build(
        document=make_document(),
        chunk=chunk,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    assert (
        result.chunking_strategy
        == "unknown"
    )

    assert (
        result.chunking_strategy_version
        == "unknown"
    )

    assert "source_metadata" in result.metadata


def test_chunking_version_is_supported_as_fallback() -> None:
    builder = KnowledgeProvenanceBuilder()

    chunk = make_chunk(
        metadata={
            "chunking_strategy": "legacy_strategy",
            "chunking_version": "0.9",
        }
    )

    result = builder.build(
        document=make_document(),
        chunk=chunk,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    assert (
        result.chunking_strategy
        == "legacy_strategy"
    )

    assert (
        result.chunking_strategy_version
        == "0.9"
    )


def test_explicit_identity_values_are_used() -> None:
    builder = KnowledgeProvenanceBuilder()

    project_id = uuid4()
    document_id = uuid4()

    result = builder.build(
        document=make_document(),
        chunk=make_chunk(),
        project_id=project_id,
        document_id=document_id,
        document_version=9,
    )

    assert (
        result.provenance.source.project_id
        == project_id
    )

    assert (
        result.provenance.document.document_id
        == document_id
    )

    assert (
        result.provenance.document.document_version
        == 9
    )


def test_page_numbers_can_contain_multiple_pages() -> None:
    builder = KnowledgeProvenanceBuilder()

    result = builder.build(
        document=make_document(),
        chunk=make_chunk(
            page_numbers=[4, 5, 6],
        ),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    assert (
        result.provenance.location.page_numbers
        == [4, 5, 6]
    )


def test_empty_optional_location_values_are_preserved() -> None:
    builder = KnowledgeProvenanceBuilder()

    result = builder.build(
        document=make_document(),
        chunk=make_chunk(
            page_numbers=[],
        ),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=1,
    )

    location = result.provenance.location

    assert location.page_numbers == []
    assert location.start_time is None
    assert location.end_time is None