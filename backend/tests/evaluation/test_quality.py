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
    EvaluationSeverity,
    EvaluationStatus,
)
from app.evaluation.quality import QualityEvaluator


def create_request(
    *,
    objective: str | None = None,
    audience: str | None = None,
    tone: str | None = None,
) -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source",
        objective=objective,
        audience=audience,
        tone=tone,
    )


def create_evaluation_request(
    content: object,
    *,
    objective: str | None = None,
    audience: str | None = None,
    tone: str | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        transformation_request=create_request(
            objective=objective,
            audience=audience,
            tone=tone,
        ),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content=content,
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_exposes_quality_check_type() -> None:
    assert (
        QualityEvaluator().check_type
        == EvaluationCheckType.QUALITY
    )


@pytest.mark.asyncio
async def test_meaningful_content_passes() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This is meaningful generated content with enough detail."
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score >= 0.5


@pytest.mark.asyncio
async def test_empty_content_fails() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request("")
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_CONTENT_EMPTY"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_none_content_fails() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(None)
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_CONTENT_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_short_content_generates_warning() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request("Short")
    )

    assert result.status == EvaluationStatus.PASSED
    assert any(
        issue.code == "ARTIFACT_0_CONTENT_TOO_SHORT"
        for issue in result.issues
    )
    assert any(
        issue.severity == EvaluationSeverity.WARNING
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_longer_content_receives_content_score() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This is sufficiently long content that should receive "
            "the meaningful content score."
        )
    )

    assert result.score is not None
    assert result.score >= 0.7


@pytest.mark.asyncio
async def test_objective_alignment_adds_score() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "The objective of improving security is addressed clearly.",
            objective="improving security",
        )
    )

    assert result.score is not None
    assert result.score >= 0.8
    assert not any(
        issue.code == "ARTIFACT_0_OBJECTIVE_ALIGNMENT_WEAK"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_weak_objective_alignment_is_reported() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This content discusses another topic in detail.",
            objective="improving security",
        )
    )

    assert any(
        issue.code == "ARTIFACT_0_OBJECTIVE_ALIGNMENT_WEAK"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_audience_signal_is_detected() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This guide is designed for developers building secure APIs.",
            audience="developers",
        )
    )

    assert not any(
        issue.code == "ARTIFACT_0_AUDIENCE_SIGNAL_WEAK"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_weak_audience_signal_is_reported() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This is general content about security practices.",
            audience="executives",
        )
    )

    assert any(
        issue.code == "ARTIFACT_0_AUDIENCE_SIGNAL_WEAK"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_tone_signal_is_detected() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This professional guide explains the security process.",
            tone="professional",
        )
    )

    assert not any(
        issue.code == "ARTIFACT_0_TONE_SIGNAL_WEAK"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_weak_tone_signal_is_reported() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "The security process is explained clearly.",
            tone="professional",
        )
    )

    assert any(
        issue.code == "ARTIFACT_0_TONE_SIGNAL_WEAK"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_structured_content_is_supported() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            {
                "title": "Security Guide",
                "summary": (
                    "A detailed guide explaining secure API practices."
                ),
                "recommendations": [
                    "Use authentication",
                    "Validate input",
                ],
            }
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score is not None


@pytest.mark.asyncio
async def test_nested_structured_content_is_supported() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            {
                "sections": [
                    {
                        "title": "Security",
                        "content": (
                            "Use strong authentication and "
                            "validate every request."
                        ),
                    }
                ]
            }
        )
    )

    assert result.score is not None
    assert result.score > 0


@pytest.mark.asyncio
async def test_multiple_artifacts_are_averaged() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content=(
                        "This is detailed content for the first artifact."
                    ),
                ),
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content=(
                        "This is detailed content for the second artifact."
                    ),
                ),
            ],
        ),
    )

    result = await QualityEvaluator().evaluate(request)

    assert result.score is not None
    assert result.score >= 0.7
    assert result.metadata["artifact_count"] == 2


@pytest.mark.asyncio
async def test_no_artifacts_fails() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[],
        ),
    )

    result = await QualityEvaluator().evaluate(request)

    assert result.status == EvaluationStatus.FAILED
    assert result.score == 0.0
    assert any(
        issue.code == "NO_ARTIFACTS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_score_is_bounded() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            (
                "This professional content is written for developers "
                "and directly addresses improving security with detailed "
                "recommendations."
            ),
            objective="improving security",
            audience="developers",
            tone="professional",
        )
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_check_metadata_is_present() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This is meaningful generated content with sufficient detail."
        )
    )

    assert len(result.checks) == 1
    assert (
        result.checks[0].check_type
        == EvaluationCheckType.QUALITY
    )
    assert (
        result.checks[0].metadata["artifact_count"]
        == 1
    )


@pytest.mark.asyncio
async def test_evaluator_metadata_is_present() -> None:
    result = await QualityEvaluator().evaluate(
        create_evaluation_request(
            "This is meaningful generated content with sufficient detail."
        )
    )

    assert (
        result.metadata["evaluator"]
        == "QualityEvaluator"
    )


@pytest.mark.asyncio
async def test_invalid_request_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="EvaluationRequest"):
        await QualityEvaluator().evaluate(
            "invalid"  # type: ignore[arg-type]
        )