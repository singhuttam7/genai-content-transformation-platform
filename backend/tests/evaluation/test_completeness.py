from __future__ import annotations

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationType,
)
from app.evaluation.completeness import CompletenessEvaluator
from app.evaluation.contracts import (
    EvaluationCheckType,
    EvaluationRequest,
    EvaluationStatus,
)


def create_request(
    transformation_type: TransformationType,
) -> TransformationRequest:
    return TransformationRequest(
        transformation_type=transformation_type,
        input="Test source content",
    )


def create_result(
    content: object,
) -> TransformationResult:
    return TransformationResult(
        status="completed",
        artifacts=[
            ArtifactEnvelope(
                artifact_type="test",
                content=content,
            )
        ],
    )


def create_evaluation_request(
    transformation_type: TransformationType,
    content: object,
) -> EvaluationRequest:
    return EvaluationRequest(
        transformation_request=create_request(
            transformation_type
        ),
        transformation_result=create_result(content),
    )


@pytest.mark.asyncio
async def test_exposes_completeness_check_type() -> None:
    assert (
        CompletenessEvaluator().check_type
        == EvaluationCheckType.COMPLETENESS
    )


@pytest.mark.asyncio
async def test_advisory_complete_content_passes() -> None:
    content = {
        "title": "Advisory",
        "summary": "Summary",
        "key_points": ["Point 1"],
        "recommendations": ["Recommendation 1"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_advisory_missing_component_fails() -> None:
    content = {
        "title": "Advisory",
        "summary": "Summary",
        "key_points": ["Point 1"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_MISSING_RECOMMENDATIONS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_executive_summary_requirements() -> None:
    content = {
        "title": "Executive Summary",
        "summary": "Summary",
        "key_findings": ["Finding"],
        "recommendations": ["Recommendation"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.EXECUTIVE_SUMMARY,
            content,
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_social_media_requirements() -> None:
    content = {
        "platform": "LinkedIn",
        "content": "Professional post",
        "hashtags": ["#AI"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.SOCIAL_MEDIA,
            content,
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_infographic_requirements() -> None:
    content = {
        "title": "Infographic",
        "sections": ["Section 1"],
        "visual_recommendations": ["Chart"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.INFOGRAPHIC,
            content,
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_presentation_requirements() -> None:
    content = {
        "slides": [{"title": "Introduction"}],
        "speaker_notes": ["Explain introduction"],
        "visual_recommendations": ["Diagram"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.PRESENTATION,
            content,
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_video_requirements() -> None:
    content = {
        "script": "Video script",
        "scenes": ["Scene 1"],
        "narration": "Narration",
        "subtitles": ["Subtitle 1"],
        "visual_recommendations": ["Visual 1"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            content,
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_missing_video_component_is_detected() -> None:
    content = {
        "script": "Video script",
        "scenes": ["Scene 1"],
        "narration": "Narration",
        "subtitles": ["Subtitle 1"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            content,
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code
        == "ARTIFACT_0_MISSING_VISUAL_RECOMMENDATIONS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_empty_component_is_detected() -> None:
    content = {
        "title": "Advisory",
        "summary": "Summary",
        "key_points": [],
        "recommendations": ["Recommendation"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_EMPTY_KEY_POINTS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_non_structured_content_fails() -> None:
    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            "plain text",
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code
        == "ARTIFACT_0_STRUCTURED_CONTENT_REQUIRED"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_no_artifacts_fails() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(
            TransformationType.ADVISORY
        ),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[],
        ),
    )

    result = await CompletenessEvaluator().evaluate(request)

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "NO_ARTIFACTS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_multiple_missing_components_are_reported() -> None:
    content = {
        "title": "Advisory",
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert len(result.issues) == 3


@pytest.mark.asyncio
async def test_check_metadata_contains_requirements() -> None:
    content = {
        "title": "Advisory",
        "summary": "Summary",
        "key_points": ["Point"],
        "recommendations": ["Recommendation"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert len(result.checks) == 1
    assert (
        result.checks[0].check_type
        == EvaluationCheckType.COMPLETENESS
    )
    assert (
        "recommendations"
        in result.checks[0].metadata["required_components"]
    )


@pytest.mark.asyncio
async def test_evaluator_metadata_is_present() -> None:
    content = {
        "title": "Advisory",
        "summary": "Summary",
        "key_points": ["Point"],
        "recommendations": ["Recommendation"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert (
        result.metadata["evaluator"]
        == "CompletenessEvaluator"
    )
    assert result.metadata["artifact_count"] == 1


@pytest.mark.asyncio
async def test_empty_string_component_is_detected() -> None:
    content = {
        "title": "",
        "summary": "Summary",
        "key_points": ["Point"],
        "recommendations": ["Recommendation"],
    }

    result = await CompletenessEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            content,
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_EMPTY_TITLE"
        for issue in result.issues
    )