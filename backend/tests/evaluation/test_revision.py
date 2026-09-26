from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.agents.transformation.validation import (
    RevisionRequest,
    TransformationRevisionAdapter,
    ValidationIssue,
)
from app.evaluation.contracts import (
    EvaluationCheck,
    EvaluationCheckType,
    EvaluationIssue,
    EvaluationSeverity,
    EvaluationResult,
    EvaluationStatus,
)
from app.evaluation.orchestrator import EvaluationOrchestrator
from app.evaluation.revision import EvaluationRevisionService


def make_request() -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source content for transformation.",
        objective="Create a useful advisory.",
        audience="General audience",
        tone="Professional",
    )


def make_result() -> TransformationResult:
    return TransformationResult(
        status=TransformationStatus.COMPLETED,
        artifacts=[
            ArtifactEnvelope(
                artifact_type="advisory",
                title="Advisory",
                content={
                    "title": "Advisory",
                    "summary": "A useful advisory summary.",
                    "key_points": [
                        "Important point",
                    ],
                    "recommendations": [
                        "Take action",
                    ],
                },
                metadata={"platform": "web"},
                provenance={"source": "source-document"},
            )
        ],
    )


def make_passed_evaluation() -> EvaluationResult:
    check = EvaluationCheck(
        check_type=EvaluationCheckType.STRUCTURE,
        status=EvaluationStatus.PASSED,
        score=1.0,
    )

    return EvaluationResult(
        status=EvaluationStatus.PASSED,
        score=1.0,
        checks=[check],
    )


def make_revision_evaluation() -> EvaluationResult:
    issue = EvaluationIssue(
        code="QUALITY_WARNING",
        message="The generated content requires revision.",
        severity=EvaluationSeverity.WARNING,
        check_type=EvaluationCheckType.QUALITY,
    )

    check = EvaluationCheck(
        check_type=EvaluationCheckType.QUALITY,
        status=EvaluationStatus.NEEDS_REVIEW,
        score=0.4,
        issues=[issue],
    )

    return EvaluationResult(
        status=EvaluationStatus.NEEDS_REVIEW,
        score=0.4,
        checks=[check],
        issues=[issue],
    )


def make_failed_evaluation() -> EvaluationResult:
    issue = EvaluationIssue(
        code="STRUCTURE_ERROR",
        message="The artifact structure is invalid.",
        severity=EvaluationSeverity.ERROR,
        check_type=EvaluationCheckType.STRUCTURE,
    )

    check = EvaluationCheck(
        check_type=EvaluationCheckType.STRUCTURE,
        status=EvaluationStatus.FAILED,
        issues=[issue],
    )

    return EvaluationResult(
        status=EvaluationStatus.FAILED,
        checks=[check],
        issues=[issue],
    )


def make_adapter() -> TransformationRevisionAdapter:
    agent = Mock()
    agent.execute = AsyncMock(
        return_value=make_result()
    )

    return TransformationRevisionAdapter(agent)


class TestEvaluationRevisionService:
    def test_default_evaluator_is_created(self):
        service = EvaluationRevisionService(
            revision_adapter=make_adapter(),
        )

        assert isinstance(
            service.evaluator,
            EvaluationOrchestrator,
        )

    def test_revision_adapter_is_preserved(self):
        adapter = make_adapter()

        service = EvaluationRevisionService(
            revision_adapter=adapter,
        )

        assert service.revision_adapter is adapter

    def test_default_max_revisions_is_one(self):
        service = EvaluationRevisionService(
            revision_adapter=make_adapter(),
        )

        assert service.max_revisions == 1

    def test_custom_max_revisions_is_preserved(self):
        service = EvaluationRevisionService(
            revision_adapter=make_adapter(),
            max_revisions=3,
        )

        assert service.max_revisions == 3

    def test_invalid_evaluator_rejected(self):
        with pytest.raises(
            TypeError,
            match="evaluator must be an EvaluationOrchestrator",
        ):
            EvaluationRevisionService(
                evaluator=Mock(),
                revision_adapter=make_adapter(),
            )

    def test_invalid_revision_adapter_rejected(self):
        with pytest.raises(
            TypeError,
            match="revision_adapter must be a "
            "TransformationRevisionAdapter",
        ):
            EvaluationRevisionService(
                revision_adapter=Mock(),
            )

    def test_negative_max_revisions_rejected(self):
        with pytest.raises(
            ValueError,
            match="max_revisions cannot be negative",
        ):
            EvaluationRevisionService(
                revision_adapter=make_adapter(),
                max_revisions=-1,
            )

    def test_non_integer_max_revisions_rejected(self):
        with pytest.raises(
            TypeError,
            match="max_revisions must be an integer",
        ):
            EvaluationRevisionService(
                revision_adapter=make_adapter(),
                max_revisions="1",
            )

    def test_boolean_max_revisions_rejected(self):
        with pytest.raises(
            TypeError,
            match="max_revisions must be an integer",
        ):
            EvaluationRevisionService(
                revision_adapter=make_adapter(),
                max_revisions=True,
            )

    @pytest.mark.asyncio
    async def test_invalid_request_rejected(self):
        service = EvaluationRevisionService(
            revision_adapter=make_adapter(),
        )

        with pytest.raises(
            TypeError,
            match="request must be a TransformationRequest",
        ):
            await service.evaluate_and_revise(
                request=Mock(),
                result=make_result(),
            )

    @pytest.mark.asyncio
    async def test_invalid_result_rejected(self):
        service = EvaluationRevisionService(
            revision_adapter=make_adapter(),
        )

        with pytest.raises(
            TypeError,
            match="result must be a TransformationResult",
        ):
            await service.evaluate_and_revise(
                request=make_request(),
                result=Mock(),
            )

    @pytest.mark.asyncio
    async def test_passed_result_is_returned_without_revision(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )
        evaluator.evaluate = AsyncMock(
            return_value=make_passed_evaluation(),
        )

        adapter = make_adapter()
        adapter.revise = AsyncMock()

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
        )

        request = make_request()
        result = make_result()

        final_result, evaluation = (
            await service.evaluate_and_revise(
                request=request,
                result=result,
            )
        )

        assert final_result is result
        assert evaluation.status == EvaluationStatus.PASSED
        evaluator.evaluate.assert_awaited_once()
        adapter.revise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_needs_review_triggers_revision(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )
        evaluator.evaluate = AsyncMock(
            side_effect=[
                make_revision_evaluation(),
                make_passed_evaluation(),
            ],
        )

        adapter = make_adapter()
        revised_result = make_result()

        adapter.revise = AsyncMock(
            return_value=revised_result,
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
        )

        final_result, evaluation = (
            await service.evaluate_and_revise(
                request=make_request(),
                result=make_result(),
            )
        )

        assert final_result is revised_result
        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluator.evaluate.await_count == 2
        adapter.revise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_evaluation_triggers_revision(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )
        evaluator.evaluate = AsyncMock(
            side_effect=[
                make_failed_evaluation(),
                make_passed_evaluation(),
            ],
        )

        adapter = make_adapter()

        adapter.revise = AsyncMock(
            return_value=make_result(),
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
        )

        _, evaluation = await service.evaluate_and_revise(
            request=make_request(),
            result=make_result(),
        )

        assert evaluation.status == EvaluationStatus.PASSED
        adapter.revise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_revision_request_preserves_original_request(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )
        evaluator.evaluate = AsyncMock(
            side_effect=[
                make_revision_evaluation(),
                make_passed_evaluation(),
            ],
        )

        adapter = make_adapter()

        adapter.revise = AsyncMock(
            return_value=make_result(),
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
        )

        request = make_request()
        result = make_result()

        await service.evaluate_and_revise(
            request=request,
            result=result,
        )

        revision_request = (
            adapter.revise.call_args.args[0]
        )

        assert isinstance(
            revision_request,
            RevisionRequest,
        )

        assert revision_request.original_request is request
        assert revision_request.previous_result is result

    @pytest.mark.asyncio
    async def test_revision_request_contains_converted_issues(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )

        evaluation = make_revision_evaluation()

        evaluator.evaluate = AsyncMock(
            side_effect=[
                evaluation,
                make_passed_evaluation(),
            ],
        )

        adapter = make_adapter()

        adapter.revise = AsyncMock(
            return_value=make_result(),
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
        )

        await service.evaluate_and_revise(
            request=make_request(),
            result=make_result(),
        )

        revision_request = (
            adapter.revise.call_args.args[0]
        )

        assert len(revision_request.issues) == 1

        issue = revision_request.issues[0]

        assert isinstance(
            issue,
            ValidationIssue,
        )

        assert issue.code == "QUALITY_WARNING"
        assert (
            issue.message
            == "The generated content requires revision."
        )
        assert issue.severity == "warning"

    @pytest.mark.asyncio
    async def test_revision_instructions_include_issue_messages(
        self,
    ):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )

        evaluator.evaluate = AsyncMock(
            side_effect=[
                make_revision_evaluation(),
                make_passed_evaluation(),
            ],
        )

        adapter = make_adapter()

        adapter.revise = AsyncMock(
            return_value=make_result(),
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
        )

        await service.evaluate_and_revise(
            request=make_request(),
            result=make_result(),
        )

        revision_request = (
            adapter.revise.call_args.args[0]
        )

        assert (
            "The generated content requires revision."
            in revision_request.instructions
        )

    @pytest.mark.asyncio
    async def test_zero_max_revisions_does_not_revise(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )

        evaluator.evaluate = AsyncMock(
            return_value=make_revision_evaluation(),
        )

        adapter = make_adapter()
        adapter.revise = AsyncMock()

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
            max_revisions=0,
        )

        result, evaluation = (
            await service.evaluate_and_revise(
                request=make_request(),
                result=make_result(),
            )
        )

        assert result is not None
        assert evaluation.status == (
            EvaluationStatus.NEEDS_REVIEW
        )

        evaluator.evaluate.assert_awaited_once()
        adapter.revise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_revision_count_is_bounded(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )

        evaluator.evaluate = AsyncMock(
            return_value=make_revision_evaluation(),
        )

        adapter = make_adapter()

        adapter.revise = AsyncMock(
            return_value=make_result(),
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
            max_revisions=1,
        )

        _, evaluation = await service.evaluate_and_revise(
            request=make_request(),
            result=make_result(),
        )

        assert evaluation.status == (
            EvaluationStatus.NEEDS_REVIEW
        )

        assert evaluator.evaluate.await_count == 2
        assert adapter.revise.await_count == 1

    @pytest.mark.asyncio
    async def test_revision_can_fail_and_stop_at_limit(self):
        evaluator = Mock(
            spec=EvaluationOrchestrator,
        )

        evaluator.evaluate = AsyncMock(
            side_effect=[
                make_revision_evaluation(),
                make_failed_evaluation(),
            ],
        )

        adapter = make_adapter()

        adapter.revise = AsyncMock(
            return_value=make_result(),
        )

        service = EvaluationRevisionService(
            evaluator=evaluator,
            revision_adapter=adapter,
            max_revisions=1,
        )

        _, evaluation = await service.evaluate_and_revise(
            request=make_request(),
            result=make_result(),
        )

        assert evaluation.status == EvaluationStatus.FAILED
        assert evaluator.evaluate.await_count == 2
        assert adapter.revise.await_count == 1