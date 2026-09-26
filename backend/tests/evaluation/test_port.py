from __future__ import annotations

import pytest

from app.agents.transformation.contracts import (
    TransformationRequest,
    TransformationResult,
)
from app.evaluation.contracts import (
    EvaluationCheckType,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
)
from app.evaluation.port import EvaluationPort


class FakeEvaluator(EvaluationPort):
    @property
    def check_type(self) -> EvaluationCheckType:
        return EvaluationCheckType.STRUCTURE

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        return EvaluationResult(
            status=EvaluationStatus.PASSED,
        )


def create_request() -> EvaluationRequest:
    transformation_request = TransformationRequest(
        transformation_type="advisory",
        input="Source",
    )

    transformation_result = TransformationResult(
        status="completed",
    )

    return EvaluationRequest(
        transformation_request=transformation_request,
        transformation_result=transformation_result,
    )


def test_evaluation_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        EvaluationPort()


def test_fake_evaluator_is_valid() -> None:
    evaluator = FakeEvaluator()

    assert isinstance(evaluator, EvaluationPort)


def test_check_type_is_exposed() -> None:
    assert (
        FakeEvaluator().check_type
        == EvaluationCheckType.STRUCTURE
    )


@pytest.mark.asyncio
async def test_evaluate_accepts_evaluation_request() -> None:
    result = await FakeEvaluator().evaluate(
        create_request()
    )

    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_evaluate_returns_evaluation_result() -> None:
    result = await FakeEvaluator().evaluate(
        create_request()
    )

    assert result.status == EvaluationStatus.PASSED