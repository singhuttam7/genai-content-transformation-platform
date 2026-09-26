from __future__ import annotations

import pytest

from app.agents.base import AgentState, AgentStatus
from app.orchestration.langgraph_adapter import LangGraphAdapter


@pytest.mark.asyncio
async def test_adapter_executes_single_node():
    async def handler(state: AgentState) -> AgentState:
        state.set_status(AgentStatus.COMPLETED)
        return state

    adapter = LangGraphAdapter(handler)

    initial_state = AgentState()

    result = await adapter.execute(initial_state)

    assert isinstance(result, AgentState)
    assert result.status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_adapter_preserves_execution_id():
    async def handler(state: AgentState) -> AgentState:
        return state

    adapter = LangGraphAdapter(handler)

    initial_state = AgentState()

    result = await adapter.execute(initial_state)

    assert result.execution_id == initial_state.execution_id


@pytest.mark.asyncio
async def test_adapter_preserves_metadata():
    async def handler(state: AgentState) -> AgentState:
        return state

    adapter = LangGraphAdapter(handler)

    initial_state = AgentState(
        metadata={
            "workflow_id": "workflow-1",
            "source": "test",
        }
    )

    result = await adapter.execute(initial_state)

    assert result.metadata == {
        "workflow_id": "workflow-1",
        "source": "test",
    }


@pytest.mark.asyncio
async def test_adapter_handler_can_update_current_agent():
    async def handler(state: AgentState) -> AgentState:
        state.set_current_agent("test-agent")
        return state

    adapter = LangGraphAdapter(handler)

    result = await adapter.execute(AgentState())

    assert result.current_agent == "test-agent"


@pytest.mark.asyncio
async def test_adapter_handler_can_update_history():
    from app.agents.base import AgentStep

    async def handler(state: AgentState) -> AgentState:
        state.add_step(
            AgentStep(
                agent_name="test-agent",
                status=AgentStatus.COMPLETED,
            )
        )
        return state

    adapter = LangGraphAdapter(handler)

    result = await adapter.execute(AgentState())

    assert len(result.history) == 1
    assert result.history[0].agent_name == "test-agent"


@pytest.mark.asyncio
async def test_adapter_rejects_non_agent_state_input():
    async def handler(state: AgentState) -> AgentState:
        return state

    adapter = LangGraphAdapter(handler)

    with pytest.raises(TypeError, match="agent_state must be an AgentState"):
        await adapter.execute({"invalid": "state"})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_adapter_rejects_invalid_handler_result():
    async def handler(state: AgentState):
        return {"invalid": "result"}

    adapter = LangGraphAdapter(handler)

    with pytest.raises(
        TypeError,
        match="LangGraph node handlers must return AgentState",
    ):
        await adapter.execute(AgentState())


@pytest.mark.asyncio
async def test_adapter_builds_compiled_graph():
    async def handler(state: AgentState) -> AgentState:
        return state

    adapter = LangGraphAdapter(handler)

    graph = adapter.build()

    assert graph is not None