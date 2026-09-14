from __future__ import annotations

import json
from uuid import UUID

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

DOCUMENT_ID = UUID(
    "33333333-3333-3333-3333-333333333333"
)

SOURCE_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)

DOCUMENT_HASH = "a" * 64
CHUNK_HASH = "b" * 64


def make_document() -> NormalizedKnowledgeDocument:
    source = SourceReference(
        source_id=SOURCE_ID,
        source_type=InputType.VIDEO,
        title="AI Research Video",
        filename="research.mp4",
        mime_type="video/mp4",
        content_hash="c" * 64,
        storage_uri="storage://research.mp4",
    )

    return NormalizedKnowledgeDocument(
        source=source,
        title="AI Research Video",
        language="en",
        text="A normalized video document.",
        elements=[],
        entities=["Entity A"],
        topics=["AI"],
        claims=["Claim A"],
        keywords=["RAG", "LLM"],
        context={
            "domain": "artificial intelligence",
        },
        provenance={
            "source_system": "ingestion",
        },
        metadata={
            "media": {
                "duration": 120.5,
                "codec": "h264",
            },
            "custom": {
                "nested": True,
            },
        },
        content_hash=DOCUMENT_HASH,
    )


def make_chunk() -> KnowledgeChunkDraft:
    return KnowledgeChunkDraft(
        chunk_index=3,
        text="This is a video knowledge chunk.",
        content_hash=CHUNK_HASH,
        token_count=7,
        source=ChunkSourceReference(
            element_orders=[10, 11],
            block_types=[
                ContentBlockType.HEADING,
                ContentBlockType.PARAGRAPH,
            ],
            page_numbers=[],
            start_time=42.5,
            end_time=58.75,
            section_path=[
                "Chapter 2",
                "Experiments",
            ],
        ),
        metadata={
            "chunking_strategy": "structure_aware",
            "chunking_strategy_version": "1.0",
            "custom": {
                "value": 123,
            },
        },
    )


def build_metadata() -> KnowledgeChunkMetadata:
    builder = KnowledgeProvenanceBuilder()

    return builder.build(
        document=make_document(),
        chunk=make_chunk(),
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        document_version=2,
    )


def test_model_dump_contains_complete_structure() -> None:
    result = build_metadata()

    data = result.model_dump()

    assert data["chunk_index"] == 3
    assert data["content_hash"] == CHUNK_HASH

    assert data["provenance"]["source"][
        "project_id"
    ] == PROJECT_ID

    assert data["provenance"]["source"][
        "source_id"
    ] == SOURCE_ID

    assert data["provenance"]["document"][
        "document_id"
    ] == DOCUMENT_ID

    assert data["provenance"]["document"][
        "document_version"
    ] == 2

    assert data["provenance"]["document"][
        "content_hash"
    ] == DOCUMENT_HASH


def test_json_serialization_is_valid() -> None:
    result = build_metadata()

    json_data = result.model_dump_json()

    parsed = json.loads(json_data)

    assert isinstance(parsed, dict)
    assert parsed["chunk_index"] == 3


def test_json_serialization_contains_uuid_strings() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    assert (
        data["provenance"]["source"]["project_id"]
        == str(PROJECT_ID)
    )

    assert (
        data["provenance"]["source"]["source_id"]
        == str(SOURCE_ID)
    )

    assert (
        data["provenance"]["document"]["document_id"]
        == str(DOCUMENT_ID)
    )


def test_json_serialization_contains_enum_values() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    block_types = data["provenance"][
        "location"
    ]["block_types"]

    assert block_types == [
        "heading",
        "paragraph",
    ]


def test_json_round_trip_reconstructs_object() -> None:
    original = build_metadata()

    json_data = original.model_dump_json()

    restored = KnowledgeChunkMetadata.model_validate_json(
        json_data
    )

    assert restored == original


def test_model_dump_json_is_deterministic() -> None:
    first = build_metadata().model_dump_json()
    second = build_metadata().model_dump_json()

    assert first == second


def test_model_dump_is_deterministic() -> None:
    first = build_metadata().model_dump()
    second = build_metadata().model_dump()

    assert first == second


def test_nested_source_metadata_survives_serialization() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    source_metadata = data["metadata"][
        "source_metadata"
    ]

    assert source_metadata["source_type"] == "video"

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


def test_nested_custom_metadata_survives_serialization() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    assert data["metadata"]["custom"] == {
        "value": 123,
    }


def test_media_metadata_survives_serialization() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    media = data["metadata"]["source_metadata"][
        "metadata"
    ]["media"]

    assert media["duration"] == 120.5
    assert media["codec"] == "h264"


def test_timestamp_provenance_survives_serialization() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    location = data["provenance"]["location"]

    assert location["start_time"] == 42.5
    assert location["end_time"] == 58.75


def test_section_provenance_survives_serialization() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    assert data["provenance"]["location"][
        "section_path"
    ] == [
        "Chapter 2",
        "Experiments",
    ]


def test_element_order_provenance_survives_serialization() -> None:
    result = build_metadata()

    data = json.loads(
        result.model_dump_json()
    )

    assert data["provenance"]["location"][
        "element_orders"
    ] == [10, 11]


def test_metadata_can_be_used_as_jsonb_payload() -> None:
    result = build_metadata()

    payload = result.model_dump(
        mode="json"
    )

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded == payload


def test_serialization_does_not_change_original_object() -> None:
    result = build_metadata()

    before = result.model_dump_json()

    result.model_dump()
    result.model_dump_json()
    result.model_dump(mode="json")

    after = result.model_dump_json()

    assert before == after