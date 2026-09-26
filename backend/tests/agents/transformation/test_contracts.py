from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.transformation import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)


class TestTransformationType:
    def test_all_transformation_types_exist(self) -> None:
        assert {
            TransformationType.ADVISORY,
            TransformationType.EXECUTIVE_SUMMARY,
            TransformationType.SOCIAL_MEDIA,
            TransformationType.INFOGRAPHIC,
            TransformationType.PRESENTATION,
            TransformationType.VIDEO,
        } == set(TransformationType)

    def test_values_are_stable(self) -> None:
        assert TransformationType.ADVISORY.value == "advisory"
        assert (
            TransformationType.EXECUTIVE_SUMMARY.value
            == "executive_summary"
        )
        assert TransformationType.SOCIAL_MEDIA.value == "social_media"
        assert TransformationType.INFOGRAPHIC.value == "infographic"
        assert TransformationType.PRESENTATION.value == "presentation"
        assert TransformationType.VIDEO.value == "video"


class TestTransformationStatus:
    def test_all_lifecycle_statuses_exist(self) -> None:
        assert {
            TransformationStatus.PENDING,
            TransformationStatus.RUNNING,
            TransformationStatus.COMPLETED,
            TransformationStatus.FAILED,
            TransformationStatus.CANCELLED,
        } == set(TransformationStatus)


class TestArtifactEnvelope:
    def test_creates_valid_artifact(self) -> None:
        artifact = ArtifactEnvelope(
            artifact_type="executive_summary",
            title="Executive Summary",
            content="Generated summary.",
        )

        assert artifact.artifact_type == "executive_summary"
        assert artifact.title == "Executive Summary"
        assert artifact.content == "Generated summary."

    def test_accepts_structured_content(self) -> None:
        artifact = ArtifactEnvelope(
            artifact_type="presentation",
            content={
                "slides": [
                    {"title": "Introduction"},
                    {"title": "Conclusion"},
                ]
            },
        )

        assert artifact.content["slides"][0]["title"] == "Introduction"

    def test_accepts_metadata(self) -> None:
        artifact = ArtifactEnvelope(
            artifact_type="advisory",
            metadata={"language": "English"},
        )

        assert artifact.metadata == {"language": "English"}

    def test_accepts_provenance(self) -> None:
        artifact = ArtifactEnvelope(
            artifact_type="summary",
            provenance={
                "source_ids": ["source-1"],
                "retrieval_ids": ["chunk-1"],
            },
        )

        assert artifact.provenance["source_ids"] == ["source-1"]

    def test_defaults_metadata_and_provenance(self) -> None:
        artifact = ArtifactEnvelope(
            artifact_type="social_media",
        )

        assert artifact.metadata == {}
        assert artifact.provenance == {}

    def test_rejects_empty_artifact_type(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactEnvelope(artifact_type="")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactEnvelope(
                artifact_type="summary",
                unsupported_field="value",
            )


class TestTransformationRequest:
    def test_creates_valid_request(self) -> None:
        request = TransformationRequest(
            transformation_type=TransformationType.ADVISORY,
            input="Source content",
        )

        assert request.transformation_type == TransformationType.ADVISORY
        assert request.input == "Source content"

    def test_accepts_all_transformation_configuration(self) -> None:
        request = TransformationRequest(
            transformation_type=TransformationType.SOCIAL_MEDIA,
            input="Source content",
            objective="Create a professional post.",
            audience="Professionals",
            tone="Professional",
            language="English",
            detail_level="detailed",
            style="LinkedIn",
            configuration={
                "platform": "linkedin",
                "max_length": 3000,
            },
            metadata={
                "request_id": "request-1",
            },
        )

        assert request.objective == "Create a professional post."
        assert request.audience == "Professionals"
        assert request.tone == "Professional"
        assert request.language == "English"
        assert request.detail_level == "detailed"
        assert request.style == "LinkedIn"
        assert request.configuration["platform"] == "linkedin"
        assert request.configuration["max_length"] == 3000
        assert request.metadata["request_id"] == "request-1"

    def test_accepts_structured_input(self) -> None:
        request = TransformationRequest(
            transformation_type=TransformationType.PRESENTATION,
            input={
                "title": "AI Transformation",
                "sections": [
                    "Introduction",
                    "Architecture",
                ],
            },
        )

        assert request.input["sections"] == [
            "Introduction",
            "Architecture",
        ]

    def test_defaults_language(self) -> None:
        request = TransformationRequest(
            transformation_type=TransformationType.VIDEO,
            input="Source",
        )

        assert request.language == "English"

    def test_defaults_configuration_and_metadata(self) -> None:
        request = TransformationRequest(
            transformation_type=TransformationType.VIDEO,
            input="Source",
        )

        assert request.configuration == {}
        assert request.metadata == {}

    def test_rejects_empty_language(self) -> None:
        with pytest.raises(ValidationError):
            TransformationRequest(
                transformation_type=TransformationType.VIDEO,
                input="Source",
                language="",
            )

    def test_rejects_invalid_transformation_type(self) -> None:
        with pytest.raises(ValidationError):
            TransformationRequest(
                transformation_type="unsupported",
                input="Source",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TransformationRequest(
                transformation_type=TransformationType.ADVISORY,
                input="Source",
                unsupported_field=True,
            )


class TestTransformationResult:
    def test_creates_completed_result_without_artifacts(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.COMPLETED,
        )

        assert result.status == TransformationStatus.COMPLETED
        assert result.artifacts == []
        assert result.error is None

    def test_creates_completed_result_with_artifact(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.COMPLETED,
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="executive_summary",
                    title="Summary",
                    content="Generated summary.",
                )
            ],
        )

        assert len(result.artifacts) == 1
        assert (
            result.artifacts[0].artifact_type
            == "executive_summary"
        )

    def test_supports_multiple_artifacts(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.COMPLETED,
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="summary",
                    content="Summary",
                ),
                ArtifactEnvelope(
                    artifact_type="social_media",
                    content="Post",
                ),
            ],
        )

        assert [
            artifact.artifact_type
            for artifact in result.artifacts
        ] == [
            "summary",
            "social_media",
        ]

    def test_accepts_metadata(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.COMPLETED,
            metadata={"model": "test-model"},
        )

        assert result.metadata == {
            "model": "test-model",
        }

    def test_failed_result_requires_error(self) -> None:
        with pytest.raises(ValueError):
            TransformationResult(
                status=TransformationStatus.FAILED,
            )

    def test_failed_result_accepts_error(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.FAILED,
            error="Transformation failed.",
        )

        assert result.error == "Transformation failed."

    def test_non_failed_result_rejects_error(self) -> None:
        with pytest.raises(ValueError):
            TransformationResult(
                status=TransformationStatus.COMPLETED,
                error="Unexpected error.",
            )

    def test_cancelled_result_does_not_require_error(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.CANCELLED,
        )

        assert result.status == TransformationStatus.CANCELLED
        assert result.error is None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TransformationResult(
                status=TransformationStatus.COMPLETED,
                unsupported_field=True,
            )

    def test_result_serialization_is_deterministic(self) -> None:
        result = TransformationResult(
            status=TransformationStatus.COMPLETED,
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="summary",
                    title="Summary",
                    content="Content",
                    metadata={
                        "language": "English",
                    },
                    provenance={
                        "source_ids": [
                            "source-1",
                        ],
                    },
                )
            ],
            metadata={
                "request_id": "request-1",
            },
        )

        assert result.model_dump() == result.model_dump()