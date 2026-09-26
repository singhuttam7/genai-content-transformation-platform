from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
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
from app.evaluation.orchestrator import EvaluationOrchestrator
from app.evaluation.port import EvaluationPort


def create_request() -> EvaluationRequest:
    return EvaluationRequest(
        transformation_request=TransformationRequest(
            transformation_type=TransformationType.ADVISORY,
            input="Source content",
        ),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content={
                        "title": "Security Advisory",
                        "summary": "Security summary",
                        "key_points": ["Point"],
                        "recommendations": ["Enable MFA"],
                    },
                    provenance={
                        "source_id": "document-001",
                    },
                )
            ],
        ),
    )


def create_evaluator(
    check_type: EvaluationCheckType,
    *,
    status: EvaluationStatus = EvaluationStatus.PASSED,
    score: float = 1.0,
) -> EvaluationPort:
    evaluator = Mock(spec=EvaluationPort)
    evaluator.check_type = check_type
    evaluator.evaluate = AsyncMock(
        return_value=EvaluationResult(
            status=status,
            score=score,
            checks=[
                EvaluationCheck(
                    check_type=check_type,
                    status=status,
                    score=score,
                )
            ],
        )
    )
    return evaluator


@pytest.mark.asyncio
async def test_default_orchestrator_contains_all_evaluators() -> None:
    orchestrator = EvaluationOrchestrator()

    assert [
        evaluator.check_type
        for evaluator in orchestrator.evaluators
    ] == [
        EvaluationCheckType.STRUCTURE,
        EvaluationCheckType.COMPLETENESS,
        EvaluationCheckType.PROVENANCE,
        EvaluationCheckType.TRANSFORMATION,
        EvaluationCheckType.QUALITY,
    ]


@pytest.mark.asyncio
async def test_all_default_checks_execute() -> None:
    result = await EvaluationOrchestrator().evaluate(
        create_request()
    )

    assert result.status == EvaluationStatus.PASSED
    assert len(result.checks) == 5
    assert result.score is not None


@pytest.mark.asyncio
async def test_only_requested_checks_execute() -> None:
    evaluators = [
        create_evaluator(EvaluationCheckType.STRUCTURE),
        create_evaluator(EvaluationCheckType.QUALITY),
    ]

    orchestrator = EvaluationOrchestrator(evaluators)

    request = create_request()
    request.checks = [
        EvaluationCheckType.QUALITY,
    ]

    result = await orchestrator.evaluate(request)

    assert result.status == EvaluationStatus.PASSED
    assert len(result.checks) == 1
    assert (
        result.checks[0].check_type
        == EvaluationCheckType.QUALITY
    )

    evaluators[0].evaluate.assert_not_awaited()
    evaluators[1].evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_scores_are_averaged() -> None:
    evaluators = [
        create_evaluator(
            EvaluationCheckType.STRUCTURE,
            score=1.0,
        ),
        create_evaluator(
            EvaluationCheckType.QUALITY,
            score=0.5,
        ),
    ]

    orchestrator = EvaluationOrchestrator(evaluators)

    request = create_request()
    request.checks = [
        EvaluationCheckType.STRUCTURE,
        EvaluationCheckType.QUALITY,
    ]

    result = await orchestrator.evaluate(request)

    assert result.score == 0.75


@pytest.mark.asyncio
async def test_failed_check_causes_failed_result() -> None:
    evaluators = [
        create_evaluator(
            EvaluationCheckType.STRUCTURE,
            status=EvaluationStatus.FAILED,
            score=0.0,
        ),
        create_evaluator(
            EvaluationCheckType.QUALITY,
            score=1.0,
        ),
    ]

    result = await EvaluationOrchestrator(
        evaluators
    ).evaluate(create_request())

    assert result.status == EvaluationStatus.FAILED


@pytest.mark.asyncio
async def test_needs_review_is_preserved() -> None:
    evaluator = create_evaluator(
        EvaluationCheckType.QUALITY,
        status=EvaluationStatus.NEEDS_REVIEW,
        score=0.4,
    )

    request = create_request()
    request.checks = [
        EvaluationCheckType.QUALITY,
    ]

    result = await EvaluationOrchestrator(
        [evaluator]
    ).evaluate(request)

    assert result.status == EvaluationStatus.NEEDS_REVIEW
    assert result.score == 0.4


@pytest.mark.asyncio
async def test_failed_takes_precedence_over_needs_review() -> None:
    evaluators = [
        create_evaluator(
            EvaluationCheckType.STRUCTURE,
            status=EvaluationStatus.NEEDS_REVIEW,
            score=0.4,
        ),
        create_evaluator(
            EvaluationCheckType.QUALITY,
            status=EvaluationStatus.FAILED,
            score=0.0,
        ),
    ]

    result = await EvaluationOrchestrator(
        evaluators
    ).evaluate(create_request())

    assert result.status == EvaluationStatus.FAILED


@pytest.mark.asyncio
async def test_issues_are_aggregated() -> None:
    issue = EvaluationIssue(
        code="QUALITY_WARNING",
        message="Quality requires review.",
        severity=EvaluationSeverity.WARNING,
        check_type=EvaluationCheckType.QUALITY,
    )

    quality_check = EvaluationCheck(
        check_type=EvaluationCheckType.QUALITY,
        status=EvaluationStatus.NEEDS_REVIEW,
        score=0.4,
        issues=[issue],
    )

    evaluator = create_evaluator(
        EvaluationCheckType.QUALITY
    )

    evaluator.evaluate = AsyncMock(
        return_value=EvaluationResult(
            status=EvaluationStatus.NEEDS_REVIEW,
            score=0.4,
            checks=[quality_check],
            issues=[issue],
        )
    )

    request = create_request()
    request.checks = [
        EvaluationCheckType.QUALITY,
    ]

    result = await EvaluationOrchestrator(
        [evaluator]
    ).evaluate(request)

    assert result.issues == [issue]
    assert result.checks == [quality_check]
    assert result.status == EvaluationStatus.NEEDS_REVIEW
    assert result.score == 0.4


@pytest.mark.asyncio
async def test_checks_are_aggregated() -> None:
    evaluators = [
        create_evaluator(EvaluationCheckType.STRUCTURE),
        create_evaluator(EvaluationCheckType.QUALITY),
    ]

    request = create_request()
    request.checks = [
        EvaluationCheckType.STRUCTURE,
        EvaluationCheckType.QUALITY,
    ]

    result = await EvaluationOrchestrator(
        evaluators
    ).evaluate(request)

    assert [
        check.check_type
        for check in result.checks
    ] == [
        EvaluationCheckType.STRUCTURE,
        EvaluationCheckType.QUALITY,
    ]


@pytest.mark.asyncio
async def test_metadata_is_present() -> None:
    evaluator = create_evaluator(
        EvaluationCheckType.QUALITY
    )

    request = create_request()
    request.checks = [
        EvaluationCheckType.QUALITY,
    ]

    result = await EvaluationOrchestrator(
        [evaluator]
    ).evaluate(request)

    assert (
        result.metadata["orchestrator"]
        == "EvaluationOrchestrator"
    )
    assert result.metadata["check_count"] == 1


@pytest.mark.asyncio
async def test_empty_evaluator_list_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="At least one evaluation evaluator",
    ):
        EvaluationOrchestrator([])


@pytest.mark.asyncio
async def test_invalid_evaluator_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="implement EvaluationPort",
    ):
        EvaluationOrchestrator(["invalid"])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_invalid_request_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="EvaluationRequest",
    ):
        await EvaluationOrchestrator().evaluate(
            "invalid"  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_no_requested_checks_are_rejected() -> None:
    evaluator = create_evaluator(
        EvaluationCheckType.QUALITY
    )

    request = create_request()
    request.checks = [
        EvaluationCheckType.STRUCTURE,
    ]

    with pytest.raises(
        ValueError,
        match="No requested evaluation checks",
    ):
        await EvaluationOrchestrator(
            [evaluator]
        ).evaluate(request)


@pytest.mark.asyncio
async def test_non_evaluation_result_is_rejected() -> None:
    evaluator = Mock(spec=EvaluationPort)
    evaluator.check_type = EvaluationCheckType.QUALITY
    evaluator.evaluate = AsyncMock(
        return_value="invalid result"
    )

    request = create_request()
    request.checks = [
        EvaluationCheckType.QUALITY,
    ]

    with pytest.raises(
        TypeError,
        match="EvaluationResult",
    ):
        await EvaluationOrchestrator(
            [evaluator]
        ).evaluate(request)