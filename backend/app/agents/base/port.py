from __future__ import annotations

from abc import ABC, abstractmethod

from app.agents.base.contracts import AgentRequest, AgentResult


class AgentPort(ABC):
    """
    Provider-independent execution contract for agents.

    Concrete agents implement this interface. The orchestration
    layer depends on AgentPort rather than on concrete agent classes.
    """

    @abstractmethod
    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        """
        Execute an agent request and return its result.
        """
        raise NotImplementedError