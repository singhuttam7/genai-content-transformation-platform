from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ingestion.schemas import ContentBlockType
from app.knowledge.provenance import (
    KnowledgeChunkMetadata,
    KnowledgeChunkProvenance,
    KnowledgeDocumentProvenance,
    KnowledgeLocationProvenance,
    KnowledgeSourceProvenance,
)


def make_source_provenance() -> KnowledgeSourceProvenance:
    return KnowledgeSourceProvenance(
        project_id=uuid4(),
        source_id=uuid4(),
        source_type="pdf",
        source_uri="file:///documents/report.pdf",
        title="Research Report",
        language="en",
    )


def make_document_provenance() -> KnowledgeDocumentProvenance:
    return KnowledgeDocumentProvenance(
        document_id=uuid4(),
        document_version=1,
        content_hash="a" * 64,
    )


def make_location_provenance() -> KnowledgeLocationProvenance:
    return KnowledgeLocationProvenance(
        element_orders=[0, 1, 2],
        block_types=[
            ContentBlockType.HEADING,
            ContentBlockType.PARAGRAPH,
        ],
        page_numbers=[1, 2],
        section_path=["Introduction"],
        start_time=10.0,
        end_time=25.0,
    )


def make_chunk_provenance() -> KnowledgeChunkProvenance:
    return KnowledgeChunkProvenance(
        source=make_source_provenance(),
        document=make_document_provenance(),
        location=make_location_provenance(),
        chunk_index=0,
    )


def test_source_provenance_contract() -> None:
    provenance = make_source_provenance()

    assert provenance.source_type == "pdf"
    assert provenance.language == "en"


def test_document_provenance_contract() -> None:
    provenance = make_document_provenance()

    assert provenance.document_version == 1
    assert len(provenance.content_hash) == 64


def test_location_provenance_contract() -> None:
    provenance = make_location_provenance()

    assert provenance.element_orders == [0, 1, 2]
    assert provenance.page_numbers == [1, 2]
    assert provenance.section_path == ["Introduction"]
    assert provenance.start_time == 10.0
    assert provenance.end_time == 25.0


def test_chunk_provenance_contract() -> None:
    provenance = make_chunk_provenance()

    assert provenance.chunk_index == 0
    assert provenance.source.source_type == "pdf"
    assert provenance.document.document_version == 1


def test_chunk_metadata_contract() -> None:
    metadata = KnowledgeChunkMetadata(
        chunk_index=0,
        character_count=250,
        token_count=60,
        content_hash="b" * 64,
        chunking_strategy="structure_aware",
        chunking_strategy_version="1.0",
        provenance=make_chunk_provenance(),
        metadata={
            "custom_field": "value",
        },
    )

    assert metadata.character_count == 250
    assert metadata.token_count == 60
    assert metadata.chunking_strategy == "structure_aware"
    assert metadata.metadata["custom_field"] == "value"


def test_source_provenance_rejects_missing_project_id() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceProvenance(
            source_id=uuid4(),
            source_type="pdf",
        )


def test_document_provenance_rejects_invalid_version() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocumentProvenance(
            document_id=uuid4(),
            document_version=0,
            content_hash="a" * 64,
        )


def test_document_provenance_rejects_invalid_hash() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDocumentProvenance(
            document_id=uuid4(),
            document_version=1,
            content_hash="invalid",
        )


def test_chunk_provenance_rejects_negative_chunk_index() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunkProvenance(
            source=make_source_provenance(),
            document=make_document_provenance(),
            location=make_location_provenance(),
            chunk_index=-1,
        )


def test_location_provenance_rejects_negative_start_time() -> None:
    with pytest.raises(ValidationError):
        KnowledgeLocationProvenance(
            start_time=-1,
        )


def test_location_provenance_rejects_negative_end_time() -> None:
    with pytest.raises(ValidationError):
        KnowledgeLocationProvenance(
            end_time=-1,
        )


def test_chunk_metadata_rejects_zero_character_count() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunkMetadata(
            chunk_index=0,
            character_count=0,
            content_hash="a" * 64,
            chunking_strategy="structure_aware",
            chunking_strategy_version="1.0",
            provenance=make_chunk_provenance(),
        )


def test_chunk_metadata_allows_missing_token_count() -> None:
    metadata = KnowledgeChunkMetadata(
        chunk_index=0,
        character_count=100,
        content_hash="a" * 64,
        chunking_strategy="structure_aware",
        chunking_strategy_version="1.0",
        provenance=make_chunk_provenance(),
    )

    assert metadata.token_count is None


def test_provenance_is_immutable() -> None:
    provenance = make_source_provenance()

    with pytest.raises(ValidationError):
        provenance.source_type = "video"


def test_metadata_is_immutable() -> None:
    metadata = KnowledgeChunkMetadata(
        chunk_index=0,
        character_count=100,
        content_hash="a" * 64,
        chunking_strategy="structure_aware",
        chunking_strategy_version="1.0",
        provenance=make_chunk_provenance(),
    )

    with pytest.raises(ValidationError):
        metadata.character_count = 200


def test_provenance_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceProvenance(
            project_id=uuid4(),
            source_id=uuid4(),
            source_type="pdf",
            unknown_field="should_fail",
        )