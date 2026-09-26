from __future__ import annotations

import pytest

from app.agents.base import (
    AgentPort,
    AgentRequest,
    AgentResult,
    AgentStatus,
)


class FakeAgent(AgentPort):
    """Concrete implementation used to verify the execution contract."""

    def __init__(self) -> None:
        self.received_request: AgentRequest | None = None

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        self.received_request = request

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "message": "Agent executed successfully.",
            },
        )


class FailingAgent(AgentPort):
    """Implementation used to verify failed execution results."""

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            error="Agent execution failed.",
        )


class TestAgentPort:
    def test_agent_port_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            AgentPort()  # type: ignore[abstract]

    def test_fake_agent_is_valid_agent_port(self) -> None:
        agent = FakeAgent()

        assert isinstance(agent, AgentPort)

    @pytest.mark.asyncio
    async def test_execute_accepts_agent_request(self) -> None:
        agent = FakeAgent()

        request = AgentRequest(
            task="summarize",
            input="Example content.",
        )

        result = await agent.execute(request)

        assert isinstance(result, AgentResult)
        assert result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_receives_original_request(self) -> None:
        agent = FakeAgent()

        request = AgentRequest(
            task="transform",
            input={
                "text": "Source content.",
            },
            metadata={
                "request_id": "req-001",
            },
        )

        await agent.execute(request)

        assert agent.received_request is request

    @pytest.mark.asyncio
    async def test_execute_returns_agent_result(self) -> None:
        agent = FakeAgent()

        request = AgentRequest(
            task="summarize",
            input="Content.",
        )

        result = await agent.execute(request)

        assert result.output == {
            "message": "Agent executed successfully.",
        }

    @pytest.mark.asyncio
    async def test_failed_execution_uses_agent_result_contract(
        self,
    ) -> None:
        agent = FailingAgent()

        request = AgentRequest(
            task="validate",
            input="Content.",
        )

        result = await agent.execute(request)

        assert isinstance(result, AgentResult)
        assert result.status == AgentStatus.FAILED
        assert result.error == "Agent execution failed."

    @pytest.mark.asyncio
    async def test_agent_port_is_provider_independent(self) -> None:
        agent = FakeAgent()

        request = AgentRequest(
            task="process",
            input={
                "provider": "test-provider",
                "model": "test-model",
            },
        )

        result = await agent.execute(request)

        assert result.status == AgentStatus.COMPLETED
        assert agent.received_request is request