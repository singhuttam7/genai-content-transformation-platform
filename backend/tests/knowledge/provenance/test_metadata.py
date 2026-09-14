from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.knowledge.provenance import (
    EnrichedKnowledgeMetadata,
    SourceMetadata,
)


def test_source_metadata_requires_source_type() -> None:
    metadata = SourceMetadata(
        source_type="pdf",
    )

    assert metadata.source_type == "pdf"
    assert metadata.metadata == {}


def test_source_metadata_preserves_metadata() -> None:
    metadata = SourceMetadata(
        source_type="pdf",
        metadata={
            "page_count": 10,
            "processor": "pdf",
        },
    )

    assert metadata.metadata == {
        "page_count": 10,
        "processor": "pdf",
    }


def test_source_metadata_accepts_nested_metadata() -> None:
    metadata = SourceMetadata(
        source_type="video",
        metadata={
            "media": {
                "duration": 120.5,
                "codec": "h264",
            },
            "scene": {
                "start": 10.0,
                "end": 20.0,
            },
        },
    )

    assert metadata.metadata["media"]["duration"] == 120.5
    assert metadata.metadata["scene"]["start"] == 10.0


def test_source_metadata_rejects_empty_source_type() -> None:
    with pytest.raises(ValidationError):
        SourceMetadata(source_type="")


def test_source_metadata_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SourceMetadata(
            source_type="pdf",
            unsupported_field="value",
        )


def test_source_metadata_is_immutable() -> None:
    metadata = SourceMetadata(
        source_type="pdf",
    )

    with pytest.raises(ValidationError):
        metadata.source_type = "video"


def test_enriched_metadata_contains_source_metadata() -> None:
    source_metadata = SourceMetadata(
        source_type="pdf",
        metadata={
            "page_count": 5,
        },
    )

    result = EnrichedKnowledgeMetadata(
        source_metadata=source_metadata,
    )

    assert result.source_metadata.source_type == "pdf"
    assert (
        result.source_metadata.metadata["page_count"]
        == 5
    )


def test_enriched_metadata_preserves_general_metadata() -> None:
    result = EnrichedKnowledgeMetadata(
        source_metadata=SourceMetadata(
            source_type="url",
            metadata={
                "canonical_url": "https://example.com",
            },
        ),
        metadata={
            "custom_field": "preserved",
            "priority": 5,
        },
    )

    assert result.metadata == {
        "custom_field": "preserved",
        "priority": 5,
    }


def test_enriched_metadata_defaults_to_empty_metadata() -> None:
    result = EnrichedKnowledgeMetadata(
        source_metadata=SourceMetadata(
            source_type="text",
        ),
    )

    assert result.metadata == {}


def test_enriched_metadata_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EnrichedKnowledgeMetadata(
            source_metadata=SourceMetadata(
                source_type="text",
            ),
            unsupported_field="value",
        )


def test_enriched_metadata_is_immutable() -> None:
    result = EnrichedKnowledgeMetadata(
        source_metadata=SourceMetadata(
            source_type="text",
        ),
    )

    with pytest.raises(ValidationError):
        result.metadata = {"changed": True}