from __future__ import annotations

from typing import Any

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.evaluation.contracts import (
    EvaluationCheckType,
    EvaluationRequest,
    EvaluationSeverity,
    EvaluationStatus,
)
from app.evaluation.structural import StructuralEvaluator


def create_request() -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Test source content",
    )


def create_result(
    *,
    status: TransformationStatus = TransformationStatus.COMPLETED,
    artifacts: list[ArtifactEnvelope] | None = None,
    error: str | None = None,
) -> TransformationResult:
    return TransformationResult(
        status=status,
        artifacts=artifacts or [],
        error=error,
    )


def create_evaluation_request(
    result: TransformationResult,
    *,
    artifacts: list[ArtifactEnvelope] | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=result,
        artifacts=artifacts or [],
    )


@pytest.mark.asyncio
async def test_exposes_structure_check_type() -> None:
    evaluator = StructuralEvaluator()

    assert evaluator.check_type == EvaluationCheckType.STRUCTURE


@pytest.mark.asyncio
async def test_valid_result_passes() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Valid advisory content",
                    )
                ]
            )
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []


@pytest.mark.asyncio
async def test_artifact_count_is_reported() -> None:
    artifacts = [
        ArtifactEnvelope(
            artifact_type="advisory",
            content="Advisory content",
        ),
        ArtifactEnvelope(
            artifact_type="executive_summary",
            content="Summary content",
        ),
    ]

    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(artifacts=artifacts)
        )
    )

    assert result.metadata["artifact_count"] == 2
    assert result.checks[0].metadata["artifact_count"] == 2


@pytest.mark.asyncio
async def test_multiple_artifacts_pass() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Advisory content",
                    ),
                    ArtifactEnvelope(
                        artifact_type="executive_summary",
                        content="Summary content",
                    ),
                    ArtifactEnvelope(
                        artifact_type="presentation",
                        content={
                            "slides": [
                                {"title": "Introduction"},
                                {"title": "Conclusion"},
                            ]
                        },
                    ),
                ]
            )
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score == 1.0
    assert len(result.checks) == 1


@pytest.mark.asyncio
async def test_missing_artifacts_fails() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result()
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.score == 0.0

    assert any(
        issue.code == "NO_ARTIFACTS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_none_content_fails() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content=None,
                    )
                ]
            )
        )
    )

    assert result.status == EvaluationStatus.FAILED

    assert any(
        issue.code == "ARTIFACT_0_CONTENT_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_blank_string_content_fails() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="   ",
                    )
                ]
            )
        )
    )

    assert result.status == EvaluationStatus.FAILED

    assert any(
        issue.code == "ARTIFACT_0_CONTENT_EMPTY"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_structured_content_is_allowed() -> None:
    structured_content: dict[str, Any] = {
        "title": "Test presentation",
        "slides": [
            {
                "number": 1,
                "title": "Introduction",
            },
            {
                "number": 2,
                "title": "Conclusion",
            },
        ],
    }

    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="presentation",
                        content=structured_content,
                    )
                ]
            )
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_artifact_type_is_required() -> None:
    artifact = ArtifactEnvelope(
        artifact_type="advisory",
        content="Content",
    )

    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[artifact]
            )
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert artifact.artifact_type == "advisory"


@pytest.mark.asyncio
async def test_failed_transformation_is_reported() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                status=TransformationStatus.FAILED,
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Error artifact",
                    )
                ],
                error="Transformation execution failed.",
            )
        )
    )

    assert result.status == EvaluationStatus.FAILED

    assert any(
        issue.code == "TRANSFORMATION_FAILED"
        for issue in result.issues
    )

    transformation_failed_issue = next(
        issue
        for issue in result.issues
        if issue.code == "TRANSFORMATION_FAILED"
    )

    assert (
        transformation_failed_issue.severity
        == EvaluationSeverity.ERROR
    )


@pytest.mark.asyncio
async def test_empty_artifact_reference_list_is_allowed() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Valid content",
                    )
                ]
            ),
            artifacts=[],
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert not any(
        issue.code == "ARTIFACT_REFERENCE_MISMATCH"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_reference_count_mismatch_is_detected() -> None:
    result_artifacts = [
        ArtifactEnvelope(
            artifact_type="advisory",
            content="Advisory content",
        ),
        ArtifactEnvelope(
            artifact_type="executive_summary",
            content="Summary content",
        ),
    ]

    referenced_artifacts = [
        ArtifactEnvelope(
            artifact_type="advisory",
            content="Advisory content",
        )
    ]

    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=result_artifacts
            ),
            artifacts=referenced_artifacts,
        )
    )

    assert result.status == EvaluationStatus.FAILED

    assert any(
        issue.code == "ARTIFACT_REFERENCE_MISMATCH"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_actual_artifact_reference_mismatch() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Advisory content",
                    )
                ]
            ),
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="presentation",
                    content="Different reference",
                ),
                ArtifactEnvelope(
                    artifact_type="video",
                    content="Another reference",
                ),
            ],
        )
    )

    assert result.status == EvaluationStatus.FAILED

    mismatch_issues = [
        issue
        for issue in result.issues
        if issue.code == "ARTIFACT_REFERENCE_MISMATCH"
    ]

    assert len(mismatch_issues) == 1


@pytest.mark.asyncio
async def test_result_contains_structure_check() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Valid content",
                    )
                ]
            )
        )
    )

    assert len(result.checks) == 1
    assert (
        result.checks[0].check_type
        == EvaluationCheckType.STRUCTURE
    )


@pytest.mark.asyncio
async def test_evaluator_metadata_is_present() -> None:
    result = await StructuralEvaluator().evaluate(
        create_evaluation_request(
            create_result(
                artifacts=[
                    ArtifactEnvelope(
                        artifact_type="advisory",
                        content="Valid content",
                    )
                ]
            )
        )
    )

    assert (
        result.metadata["evaluator"]
        == "StructuralEvaluator"
    )

    assert result.metadata["artifact_count"] == 1


@pytest.mark.asyncio
async def test_invalid_request_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="EvaluationRequest"):
        await StructuralEvaluator().evaluate(
            "invalid request"  # type: ignore[arg-type]
        )