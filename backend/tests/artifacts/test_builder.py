from __future__ import annotations

from uuid import uuid4

import pytest

from app.agents.transformation import ArtifactEnvelope
from app.artifacts import ArtifactBuildData, ArtifactBuilder


class TestArtifactBuilder:
    def test_builds_string_artifact(self) -> None:
        transformation_id = uuid4()
        execution_id = uuid4()

        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="summary",
                title="Executive Summary",
                content="Generated summary.",
            ),
            transformation_id=transformation_id,
            execution_id=execution_id,
        )

        assert isinstance(result, ArtifactBuildData)
        assert result.transformation_id == transformation_id
        assert result.execution_id == execution_id
        assert result.artifact_type == "summary"
        assert result.title == "Executive Summary"
        assert result.content == "Generated summary."
        assert result.status == "GENERATED"
        assert result.storage_uri is None
        assert result.content_hash is not None

    def test_builds_structured_artifact(self) -> None:
        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="presentation",
                content={
                    "slides": [
                        {"title": "Introduction"},
                        {"title": "Conclusion"},
                    ]
                },
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
        )

        assert result.content == (
            '{"slides":[{"title":"Introduction"},'
            '{"title":"Conclusion"}]}'
        )

    def test_structured_content_is_serialized_deterministically(
        self,
    ) -> None:
        envelope_a = ArtifactEnvelope(
            artifact_type="test",
            content={
                "b": 2,
                "a": 1,
            },
        )

        envelope_b = ArtifactEnvelope(
            artifact_type="test",
            content={
                "a": 1,
                "b": 2,
            },
        )

        builder = ArtifactBuilder()

        result_a = builder.build(
            envelope=envelope_a,
            transformation_id=uuid4(),
            execution_id=uuid4(),
        )

        result_b = builder.build(
            envelope=envelope_b,
            transformation_id=uuid4(),
            execution_id=uuid4(),
        )

        assert result_a.content == result_b.content
        assert result_a.content_hash == result_b.content_hash

    def test_content_hash_is_sha256(self) -> None:
        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="summary",
                content="hello",
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
        )

        assert result.content_hash == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    def test_none_content_has_no_hash(self) -> None:
        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="empty",
                content=None,
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
        )

        assert result.content is None
        assert result.content_hash is None

    def test_metadata_is_copied(self) -> None:
        metadata = {
            "language": "English",
            "model": "test-model",
        }

        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="summary",
                content="Content",
                metadata=metadata,
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
        )

        assert result.metadata == metadata

        metadata["new"] = "value"

        assert "new" not in result.metadata

    def test_custom_status_is_preserved(self) -> None:
        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="summary",
                content="Content",
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
            status="READY",
        )

        assert result.status == "READY"

    def test_custom_status_is_normalized(self) -> None:
        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="summary",
                content="Content",
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
            status="  READY  ",
        )

        assert result.status == "READY"

    def test_storage_uri_is_preserved(self) -> None:
        result = ArtifactBuilder().build(
            envelope=ArtifactEnvelope(
                artifact_type="video",
                content="Video metadata",
            ),
            transformation_id=uuid4(),
            execution_id=uuid4(),
            storage_uri="s3://bucket/video.mp4",
        )

        assert result.storage_uri == "s3://bucket/video.mp4"

    def test_rejects_invalid_envelope(self) -> None:
        with pytest.raises(TypeError):
            ArtifactBuilder().build(
                envelope="invalid",  # type: ignore[arg-type]
                transformation_id=uuid4(),
                execution_id=uuid4(),
            )

    def test_rejects_invalid_transformation_id(self) -> None:
        with pytest.raises(TypeError):
            ArtifactBuilder().build(
                envelope=ArtifactEnvelope(
                    artifact_type="summary",
                    content="Content",
                ),
                transformation_id="invalid",  # type: ignore[arg-type]
                execution_id=uuid4(),
            )

    def test_rejects_invalid_execution_id(self) -> None:
        with pytest.raises(TypeError):
            ArtifactBuilder().build(
                envelope=ArtifactEnvelope(
                    artifact_type="summary",
                    content="Content",
                ),
                transformation_id=uuid4(),
                execution_id="invalid",  # type: ignore[arg-type]
            )

    def test_rejects_empty_status(self) -> None:
        with pytest.raises(ValueError):
            ArtifactBuilder().build(
                envelope=ArtifactEnvelope(
                    artifact_type="summary",
                    content="Content",
                ),
                transformation_id=uuid4(),
                execution_id=uuid4(),
                status="   ",
            )