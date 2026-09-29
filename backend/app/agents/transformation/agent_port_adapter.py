from __future__ import annotations

from typing import Any

from app.agents.base import AgentPort, AgentRequest, AgentResult, AgentStatus
from app.agents.transformation.contracts import (
    TransformationRequest,
    TransformationResult,
)
from app.agents.transformation.port import TransformationAgentPort


class TransformationAgentPortAdapter(AgentPort):
    """
    Adapt a domain-specific TransformationAgentPort to the generic
    A7 AgentPort contract expected by WorkflowExecutor and AgentRegistry.
    """

    def __init__(
        self,
        agent: TransformationAgentPort,
    ) -> None:
        if not isinstance(agent, TransformationAgentPort):
            raise TypeError(
                "agent must be a TransformationAgentPort.",
            )

        self._agent = agent

    @property
    def agent(self) -> TransformationAgentPort:
        return self._agent

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        if not isinstance(request, AgentRequest):
            raise TypeError(
                "request must be an AgentRequest.",
            )

        metadata = dict(request.metadata)

        transformation_request = TransformationRequest(
            transformation_type=self._agent.transformation_type,
            input=request.input,
            objective=self._get_string(metadata, "objective"),
            audience=self._get_string(metadata, "audience"),
            tone=self._get_string(metadata, "tone"),
            language=self._get_string(
                metadata,
                "language",
            ) or "English",
            detail_level=self._get_string(
                metadata,
                "detail_level",
            ),
            style=self._get_string(metadata, "style"),
            configuration=self._get_dict(
                metadata,
                "configuration",
            ),
            metadata=metadata,
        )

        try:
            result = await self._agent.execute(
                transformation_request,
            )
        except Exception as exc:
            return AgentResult(
                status=AgentStatus.FAILED,
                error=str(exc),
                metadata={
                    "transformation_type": (
                        self._agent.transformation_type.value
                    ),
                },
            )

        return self._to_agent_result(result)

    @staticmethod
    def _get_string(
        metadata: dict[str, Any],
        key: str,
    ) -> str | None:
        value = metadata.get(key)

        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(
                f"Agent metadata '{key}' must be a string.",
            )

        return value

    @staticmethod
    def _get_dict(
        metadata: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        value = metadata.get(key)

        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError(
                f"Agent metadata '{key}' must be an object.",
            )

        return dict(value)

    @staticmethod
    def _to_agent_result(
        result: TransformationResult,
    ) -> AgentResult:
        if not isinstance(result, TransformationResult):
            raise TypeError(
                "Transformation agent returned an invalid result.",
            )

        if result.status.value == "completed":
            status = AgentStatus.COMPLETED
        elif result.status.value == "failed":
            status = AgentStatus.FAILED
        elif result.status.value == "cancelled":
            status = AgentStatus.CANCELLED
        else:
            status = AgentStatus.FAILED

        return AgentResult(
            status=status,
            output=result.artifacts,
            metadata={
                **result.metadata,
                "artifact_count": len(result.artifacts),
            },
            error=result.error,
        )