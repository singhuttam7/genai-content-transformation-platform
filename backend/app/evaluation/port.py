from __future__ import annotations

from abc import ABC, abstractmethod

from app.evaluation.contracts import (
    EvaluationCheckType,
    EvaluationRequest,
    EvaluationResult,
)


class EvaluationPort(ABC):
    """
    Provider-independent contract for evaluation components.

    Concrete evaluators implement one evaluation check type while
    the orchestration layer remains independent of the underlying
    validation implementation.
    """

    @property
    @abstractmethod
    def check_type(self) -> EvaluationCheckType:
        raise NotImplementedError

    @abstractmethod
    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        raise NotImplementedError