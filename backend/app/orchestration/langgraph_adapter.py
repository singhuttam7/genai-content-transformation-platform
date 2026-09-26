from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.base import AgentState


class LangGraphState(TypedDict):
    """Internal state representation used by LangGraph."""

    agent_state: AgentState


NodeHandler = Callable[[AgentState], Awaitable[AgentState]]


class LangGraphAdapter:
    """
    Adapter between the project's domain AgentState and LangGraph.

    LangGraph remains an orchestration implementation detail. The rest of
    the application continues to work with the domain-level AgentState.
    """

    def __init__(
        self,
        node_handler: NodeHandler,
    ) -> None:
        self._node_handler = node_handler

    async def _run_node(
        self,
        state: LangGraphState,
    ) -> LangGraphState:
        agent_state = state["agent_state"]
        updated_state = await self._node_handler(agent_state)

        if not isinstance(updated_state, AgentState):
            raise TypeError(
                "LangGraph node handlers must return AgentState."
            )

        return {
            "agent_state": updated_state,
        }

    def build(self) -> Any:
        """
        Build and compile the LangGraph workflow.

        The current foundation contains one orchestration node:

            START -> execute -> END
        """

        graph = StateGraph(LangGraphState)

        graph.add_node("execute", self._run_node)

        graph.add_edge(START, "execute")
        graph.add_edge("execute", END)

        return graph.compile()

    async def execute(
        self,
        agent_state: AgentState,
    ) -> AgentState:
        """
        Execute the compiled LangGraph workflow using domain AgentState.
        """

        if not isinstance(agent_state, AgentState):
            raise TypeError("agent_state must be an AgentState.")

        graph = self.build()

        result = await graph.ainvoke(
            {
                "agent_state": agent_state,
            }
        )

        updated_state = result["agent_state"]

        if not isinstance(updated_state, AgentState):
            raise TypeError(
                "LangGraph execution must return AgentState."
            )

        return updated_state