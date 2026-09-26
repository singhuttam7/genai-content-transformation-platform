from __future__ import annotations

from app.evaluation.completeness import CompletenessEvaluator
from app.evaluation.contracts import (
    EvaluationCheck,
    EvaluationIssue,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
)
from app.evaluation.port import EvaluationPort
from app.evaluation.provenance import ProvenanceEvaluator
from app.evaluation.quality import QualityEvaluator
from app.evaluation.structural import StructuralEvaluator
from app.evaluation.transformation_rules import (
    TransformationRulesEvaluator,
)


class EvaluationOrchestrator:
    """Run configured evaluation checks and aggregate their results."""

    def __init__(
        self,
        evaluators: list[EvaluationPort] | None = None,
    ) -> None:
        if evaluators is None:
            evaluators = [
                StructuralEvaluator(),
                CompletenessEvaluator(),
                ProvenanceEvaluator(),
                TransformationRulesEvaluator(),
                QualityEvaluator(),
            ]

        if not evaluators:
            raise ValueError(
                "At least one evaluation evaluator is required."
            )

        for evaluator in evaluators:
            if not isinstance(evaluator, EvaluationPort):
                raise TypeError(
                    "All evaluators must implement EvaluationPort."
                )

        self._evaluators = tuple(evaluators)

    @property
    def evaluators(self) -> tuple[EvaluationPort, ...]:
        return self._evaluators

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        if not isinstance(request, EvaluationRequest):
            raise TypeError(
                "request must be an EvaluationRequest."
            )

        requested_checks = set(request.checks)

        checks: list[EvaluationCheck] = []
        issues: list[EvaluationIssue] = []
        scores: list[float] = []

        for evaluator in self._evaluators:
            if evaluator.check_type not in requested_checks:
                continue

            result = await evaluator.evaluate(request)

            if not isinstance(result, EvaluationResult):
                raise TypeError(
                    "Evaluation evaluators must return "
                    "EvaluationResult instances."
                )

            checks.extend(result.checks)
            issues.extend(result.issues)

            if result.score is not None:
                scores.append(result.score)

        if not checks:
            raise ValueError(
                "No requested evaluation checks were executed."
            )

        score = (
            round(sum(scores) / len(scores), 4)
            if scores
            else None
        )

        if any(
            check.status == EvaluationStatus.FAILED
            for check in checks
        ):
            status = EvaluationStatus.FAILED
        elif any(
            check.status == EvaluationStatus.NEEDS_REVIEW
            for check in checks
        ):
            status = EvaluationStatus.NEEDS_REVIEW
        else:
            status = EvaluationStatus.PASSED

        return EvaluationResult(
            status=status,
            score=score,
            checks=checks,
            issues=issues,
            metadata={
                "orchestrator": self.__class__.__name__,
                "executed_checks": [
                    check.check_type.value
                    for check in checks
                ],
                "check_count": len(checks),
            },
        )