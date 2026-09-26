from __future__ import annotations

from abc import ABC, abstractmethod

from app.agents.transformation.contracts import (
    TransformationRequest,
    TransformationResult,
    TransformationType,
)


class TransformationAgentPort(ABC):
    """
    Provider-independent execution contract for transformation agents.

    Transformation agents operate on transformation-specific domain
    contracts rather than the generic A7 AgentRequest/AgentResult contracts.

    Integration with A7 AgentPort is intentionally handled through an
    adapter so that the A7 contract remains unchanged.
    """

    @property
    @abstractmethod
    def transformation_type(self) -> TransformationType:
        """
        Return the transformation type implemented by this agent.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self,
        request: TransformationRequest,
    ) -> TransformationResult:
        """
        Execute a transformation request.
        """
        raise NotImplementedError