from __future__ import annotations

from abc import ABC, abstractmethod

from app.tools.base.contracts import ToolRequest, ToolResult


class ToolPort(ABC):
    """
    Provider-independent execution contract for tools.

    Concrete tools implement this interface. Agents and
    orchestration depend on ToolPort rather than concrete
    tool implementations.
    """

    @abstractmethod
    async def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        """
        Execute a tool request and return its result.
        """
        raise NotImplementedError