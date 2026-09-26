from __future__ import annotations

from app.agents.transformation.contracts import (
    TransformationRequest,
    TransformationResult,
)
from app.agents.transformation.validation import (
    RevisionRequest,
    TransformationRevisionAdapter,
    ValidationIssue,
)
from app.evaluation.contracts import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
)
from app.evaluation.orchestrator import EvaluationOrchestrator


class EvaluationRevisionService:
    """Evaluate a transformation and perform bounded revisions."""

    def __init__(
        self,
        *,
        evaluator: EvaluationOrchestrator | None = None,
        revision_adapter: TransformationRevisionAdapter,
        max_revisions: int = 1,
    ) -> None:
        if evaluator is not None and not isinstance(
            evaluator,
            EvaluationOrchestrator,
        ):
            raise TypeError(
                "evaluator must be an EvaluationOrchestrator."
            )

        if not isinstance(
            revision_adapter,
            TransformationRevisionAdapter,
        ):
            raise TypeError(
                "revision_adapter must be a "
                "TransformationRevisionAdapter."
            )

        if not isinstance(max_revisions, int) or isinstance(
            max_revisions,
            bool,
        ):
            raise TypeError(
                "max_revisions must be an integer."
            )

        if max_revisions < 0:
            raise ValueError(
                "max_revisions cannot be negative."
            )

        self._evaluator = evaluator or EvaluationOrchestrator()
        self._revision_adapter = revision_adapter
        self._max_revisions = max_revisions

    @property
    def evaluator(self) -> EvaluationOrchestrator:
        return self._evaluator

    @property
    def revision_adapter(
        self,
    ) -> TransformationRevisionAdapter:
        return self._revision_adapter

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate_and_revise(
        self,
        *,
        request: TransformationRequest,
        result: TransformationResult,
    ) -> tuple[TransformationResult, EvaluationResult]:
        if not isinstance(request, TransformationRequest):
            raise TypeError(
                "request must be a TransformationRequest."
            )

        if not isinstance(result, TransformationResult):
            raise TypeError(
                "result must be a TransformationResult."
            )

        current_request = request
        current_result = result

        for revision_number in range(
            self._max_revisions + 1
        ):
            evaluation_request = EvaluationRequest(
                transformation_request=current_request,
                transformation_result=current_result,
                artifacts=current_result.artifacts,
            )

            evaluation = await self._evaluator.evaluate(
                evaluation_request
            )

            if evaluation.status == EvaluationStatus.PASSED:
                return current_result, evaluation

            if evaluation.status not in (
                EvaluationStatus.NEEDS_REVIEW,
                EvaluationStatus.FAILED,
            ):
                return current_result, evaluation

            if revision_number >= self._max_revisions:
                return current_result, evaluation

            instructions = self._build_revision_instructions(
                evaluation
            )

            revision_request = RevisionRequest(
                original_request=current_request,
                previous_result=current_result,
                instructions=instructions,
                issues=[
                    self._to_validation_issue(issue)
                    for issue in evaluation.issues
                ],
            )

            current_result = (
                await self._revision_adapter.revise(
                    revision_request
                )
            )

        return current_result, evaluation

    @staticmethod
    def _to_validation_issue(issue) -> ValidationIssue:
        """Convert an evaluation issue to the A8 validation contract."""
        return ValidationIssue(
            code=issue.code,
            message=issue.message,
            severity=issue.severity.value,
        )

    @staticmethod
    def _build_revision_instructions(
        evaluation: EvaluationResult,
    ) -> str:
        messages = [
            issue.message.strip()
            for issue in evaluation.issues
            if issue.message.strip()
        ]

        if messages:
            return (
                "Revise the transformation to address the "
                "following evaluation issues:\n"
                + "\n".join(
                    f"- {message}"
                    for message in messages
                )
            )

        return (
            "Revise the transformation to address the "
            "evaluation findings and improve the generated "
            "artifact."
        )