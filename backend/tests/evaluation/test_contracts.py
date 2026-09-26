from __future__ import annotations

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.evaluation.contracts import (
    EvaluationCheck,
    EvaluationCheckType,
    EvaluationIssue,
    EvaluationRequest,
    EvaluationResult,
    EvaluationSeverity,
    EvaluationStatus,
)


def create_request() -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source material",
    )


def create_result() -> TransformationResult:
    return TransformationResult(
        status=TransformationStatus.COMPLETED,
        artifacts=[
            ArtifactEnvelope(
                artifact_type="advisory",
                content="Generated content",
            )
        ],
    )


def create_issue(
    *,
    severity: EvaluationSeverity = EvaluationSeverity.ERROR,
) -> EvaluationIssue:
    return EvaluationIssue(
        code="TEST_ISSUE",
        message="Test evaluation issue.",
        severity=severity,
        check_type=EvaluationCheckType.STRUCTURE,
    )


def test_evaluation_status_values() -> None:
    assert EvaluationStatus.PASSED.value == "passed"
    assert EvaluationStatus.FAILED.value == "failed"
    assert (
        EvaluationStatus.NEEDS_REVIEW.value
        == "needs_review"
    )


def test_evaluation_severity_values() -> None:
    assert EvaluationSeverity.INFO.value == "info"
    assert EvaluationSeverity.WARNING.value == "warning"
    assert EvaluationSeverity.ERROR.value == "error"


def test_all_evaluation_check_types_exist() -> None:
    assert set(EvaluationCheckType) == {
        EvaluationCheckType.STRUCTURE,
        EvaluationCheckType.COMPLETENESS,
        EvaluationCheckType.PROVENANCE,
        EvaluationCheckType.FACTUAL,
        EvaluationCheckType.TRANSFORMATION,
        EvaluationCheckType.QUALITY,
    }


def test_issue_defaults_to_error() -> None:
    issue = create_issue()

    assert issue.severity == EvaluationSeverity.ERROR
    assert issue.metadata == {}


def test_issue_accepts_metadata() -> None:
    issue = EvaluationIssue(
        code="TEST",
        message="Problem",
        severity=EvaluationSeverity.WARNING,
        check_type=EvaluationCheckType.QUALITY,
        metadata={"score": 0.5},
    )

    assert issue.metadata["score"] == 0.5


def test_issue_rejects_blank_code() -> None:
    with pytest.raises(ValueError):
        EvaluationIssue(
            code="",
            message="Problem",
            check_type=EvaluationCheckType.QUALITY,
        )


def test_issue_rejects_blank_message() -> None:
    with pytest.raises(ValueError):
        EvaluationIssue(
            code="TEST",
            message="",
            check_type=EvaluationCheckType.QUALITY,
        )


def test_check_defaults() -> None:
    check = EvaluationCheck(
        check_type=EvaluationCheckType.STRUCTURE,
        status=EvaluationStatus.PASSED,
    )

    assert check.score is None
    assert check.issues == []
    assert check.metadata == {}


def test_check_accepts_score() -> None:
    check = EvaluationCheck(
        check_type=EvaluationCheckType.QUALITY,
        status=EvaluationStatus.PASSED,
        score=0.95,
    )

    assert check.score == 0.95


@pytest.mark.parametrize(
    "score",
    [-0.1, 1.1],
)
def test_check_rejects_invalid_score(
    score: float,
) -> None:
    with pytest.raises(ValueError):
        EvaluationCheck(
            check_type=EvaluationCheckType.QUALITY,
            status=EvaluationStatus.PASSED,
            score=score,
        )


def test_evaluation_request_defaults_to_all_checks() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=create_result(),
    )

    assert request.checks == list(EvaluationCheckType)


def test_evaluation_request_accepts_selected_checks() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=create_result(),
        checks=[
            EvaluationCheckType.STRUCTURE,
            EvaluationCheckType.COMPLETENESS,
        ],
    )

    assert request.checks == [
        EvaluationCheckType.STRUCTURE,
        EvaluationCheckType.COMPLETENESS,
    ]


def test_evaluation_request_accepts_artifacts() -> None:
    artifact = ArtifactEnvelope(
        artifact_type="advisory",
        content="Content",
    )

    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=create_result(),
        artifacts=[artifact],
    )

    assert request.artifacts == [artifact]


def test_evaluation_request_accepts_configuration() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=create_result(),
        configuration={
            "minimum_score": 0.8,
        },
    )

    assert request.configuration["minimum_score"] == 0.8


def test_evaluation_result_defaults() -> None:
    result = EvaluationResult(
        status=EvaluationStatus.PASSED,
    )

    assert result.score is None
    assert result.checks == []
    assert result.issues == []


def test_evaluation_result_accepts_score() -> None:
    result = EvaluationResult(
        status=EvaluationStatus.PASSED,
        score=0.9,
    )

    assert result.score == 0.9


def test_passed_result_rejects_error_issue() -> None:
    with pytest.raises(
        ValueError,
        match="error-severity issues",
    ):
        EvaluationResult(
            status=EvaluationStatus.PASSED,
            issues=[create_issue()],
        )


def test_passed_result_allows_warning() -> None:
    result = EvaluationResult(
        status=EvaluationStatus.PASSED,
        issues=[
            create_issue(
                severity=EvaluationSeverity.WARNING
            )
        ],
    )

    assert result.status == EvaluationStatus.PASSED


def test_failed_result_accepts_error_issue() -> None:
    result = EvaluationResult(
        status=EvaluationStatus.FAILED,
        issues=[create_issue()],
    )

    assert result.status == EvaluationStatus.FAILED


def test_needs_review_accepts_warning() -> None:
    result = EvaluationResult(
        status=EvaluationStatus.NEEDS_REVIEW,
        issues=[
            create_issue(
                severity=EvaluationSeverity.WARNING
            )
        ],
    )

    assert result.status == EvaluationStatus.NEEDS_REVIEW


def test_result_rejects_invalid_score() -> None:
    with pytest.raises(ValueError):
        EvaluationResult(
            status=EvaluationStatus.PASSED,
            score=1.5,
        )


def test_result_serialization_is_deterministic() -> None:
    result = EvaluationResult(
        status=EvaluationStatus.PASSED,
        score=0.9,
        metadata={
            "source": "test",
            "version": 1,
        },
    )

    first = result.model_dump_json()
    second = result.model_dump_json()

    assert first == second