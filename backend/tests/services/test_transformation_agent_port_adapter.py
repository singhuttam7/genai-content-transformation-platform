from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.base import AgentRequest, AgentResult, AgentStatus
from app.agents.transformation.agent_port_adapter import (
    TransformationAgentPortAdapter,
)
from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationResult,
    TransformationStatus,
)
from app.agents.transformation.port import TransformationAgentPort


def test_adapter_requires_transformation_agent():
    with pytest.raises(TypeError):
        TransformationAgentPortAdapter(MagicMock())


def test_adapter_is_generic_agent_port():
    agent = MagicMock(spec=TransformationAgentPort)

    adapter = TransformationAgentPortAdapter(agent)

    assert adapter.agent is agent


@pytest.mark.asyncio
async def test_adapter_executes_transformation_agent():
    agent = MagicMock(spec=TransformationAgentPort)
    agent.transformation_type = "executive_summary"

    transformation_result = TransformationResult(
        status=TransformationStatus.COMPLETED,
        artifacts=[
            ArtifactEnvelope(
                artifact_type="executive_summary",
                content="Generated summary",
            )
        ],
        metadata={},
    )

    agent.execute = AsyncMock(return_value=transformation_result)

    adapter = TransformationAgentPortAdapter(agent)

    request = AgentRequest(
        task="Transform the source content into an executive summary.",
        input="Source content",
        metadata={
            "objective": "Summarize the source.",
            "audience": "Executives",
            "tone": "Professional",
            "language": "English",
            "detail_level": "balanced",
            "style": "Structured",
            "configuration": {},
        },
    )

    result = await adapter.execute(request)

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.COMPLETED
    assert result.output == transformation_result.artifacts
    assert result.metadata["artifact_count"] == 1

    agent.execute.assert_awaited_once()