from __future__ import annotations

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationType,
)
from app.evaluation.contracts import (
    EvaluationCheckType,
    EvaluationRequest,
    EvaluationStatus,
)
from app.evaluation.transformation_rules import (
    TransformationRulesEvaluator,
)


def create_request(
    transformation_type: TransformationType,
) -> TransformationRequest:
    return TransformationRequest(
        transformation_type=transformation_type,
        input="Test source",
    )


def create_evaluation_request(
    transformation_type: TransformationType,
    content: dict,
) -> EvaluationRequest:
    return EvaluationRequest(
        transformation_request=create_request(
            transformation_type
        ),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type=transformation_type.value,
                    content=content,
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_exposes_transformation_check_type() -> None:
    assert (
        TransformationRulesEvaluator().check_type
        == EvaluationCheckType.TRANSFORMATION
    )


@pytest.mark.asyncio
async def test_advisory_rules_pass() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            {
                "title": "Security Advisory",
                "recommendations": ["Enable MFA"],
            },
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_advisory_requires_title() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            {
                "recommendations": ["Enable MFA"],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_MISSING_TITLE"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_advisory_requires_recommendation() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            {
                "title": "Security Advisory",
                "recommendations": [],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_ADVISORY_NO_RECOMMENDATIONS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_executive_summary_rules_pass() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.EXECUTIVE_SUMMARY,
            {
                "summary": "Executive summary",
                "key_findings": ["Finding"],
            },
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_executive_summary_requires_summary() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.EXECUTIVE_SUMMARY,
            {
                "key_findings": ["Finding"],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_MISSING_SUMMARY"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_executive_summary_requires_finding() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.EXECUTIVE_SUMMARY,
            {
                "summary": "Summary",
                "key_findings": [],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SUMMARY_NO_FINDINGS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_social_media_rules_pass() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.SOCIAL_MEDIA,
            {
                "platform": "LinkedIn",
                "content": "This is a meaningful professional post.",
            },
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_social_media_requires_platform() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.SOCIAL_MEDIA,
            {
                "content": "This is a meaningful professional post.",
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SOCIAL_PLATFORM_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_social_media_short_content_fails() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.SOCIAL_MEDIA,
            {
                "platform": "LinkedIn",
                "content": "Short",
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SOCIAL_CONTENT_TOO_SHORT"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_infographic_rules_pass() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.INFOGRAPHIC,
            {
                "sections": ["Overview"],
                "visual_recommendations": ["Chart"],
            },
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_infographic_requires_sections() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.INFOGRAPHIC,
            {
                "sections": [],
                "visual_recommendations": ["Chart"],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_INFOGRAPHIC_NO_SECTIONS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_infographic_requires_visual_recommendations() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.INFOGRAPHIC,
            {
                "sections": ["Overview"],
                "visual_recommendations": [],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code
        == "ARTIFACT_0_INFOGRAPHIC_NO_VISUAL_RECOMMENDATIONS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_presentation_rules_pass() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.PRESENTATION,
            {
                "slides": [
                    {"title": "Introduction"},
                    {"title": "Conclusion"},
                ],
                "speaker_notes": ["Explain"],
            },
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_presentation_requires_multiple_slides() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.PRESENTATION,
            {
                "slides": [{"title": "Introduction"}],
                "speaker_notes": ["Explain"],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_PRESENTATION_TOO_FEW_SLIDES"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_presentation_requires_speaker_notes() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.PRESENTATION,
            {
                "slides": [
                    {"title": "Introduction"},
                    {"title": "Conclusion"},
                ],
                "speaker_notes": [],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_PRESENTATION_NO_NOTES"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_video_rules_pass() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            {
                "scenes": ["Scene 1"],
                "narration": "Narration",
                "subtitles": ["Subtitle 1"],
            },
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_video_requires_scenes() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            {
                "scenes": [],
                "narration": "Narration",
                "subtitles": ["Subtitle"],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_VIDEO_NO_SCENES"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_video_requires_narration() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            {
                "scenes": ["Scene"],
                "narration": "",
                "subtitles": ["Subtitle"],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_VIDEO_NO_NARRATION"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_video_requires_subtitles() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            {
                "scenes": ["Scene"],
                "narration": "Narration",
                "subtitles": [],
            },
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_VIDEO_NO_SUBTITLES"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_structured_content_is_required() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(
            TransformationType.ADVISORY
        ),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content="plain text",
                )
            ],
        ),
    )

    result = await TransformationRulesEvaluator().evaluate(request)

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code
        == "ARTIFACT_0_STRUCTURED_CONTENT_REQUIRED"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_evaluator_metadata_is_present() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.VIDEO,
            {
                "scenes": ["Scene"],
                "narration": "Narration",
                "subtitles": ["Subtitle"],
            },
        )
    )

    assert result.metadata["evaluator"] == (
        "TransformationRulesEvaluator"
    )
    assert result.metadata["transformation_type"] == "video"


@pytest.mark.asyncio
async def test_check_metadata_contains_artifact_count() -> None:
    result = await TransformationRulesEvaluator().evaluate(
        create_evaluation_request(
            TransformationType.ADVISORY,
            {
                "title": "Advisory",
                "recommendations": ["Recommendation"],
            },
        )
    )

    assert len(result.checks) == 1
    assert (
        result.checks[0].check_type
        == EvaluationCheckType.TRANSFORMATION
    )
    assert result.checks[0].metadata["artifact_count"] == 1