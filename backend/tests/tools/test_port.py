from __future__ import annotations

import pytest

from app.tools.base import (
    ToolPort,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class FakeTool(ToolPort):
    """Concrete implementation used to verify the tool contract."""

    def __init__(self) -> None:
        self.received_request: ToolRequest | None = None

    async def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        self.received_request = request

        return ToolResult(
            status=ToolStatus.COMPLETED,
            output={
                "message": "Tool executed successfully.",
            },
        )


class FailingTool(ToolPort):
    """Implementation used to verify failed tool results."""

    async def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        return ToolResult(
            status=ToolStatus.FAILED,
            error="Tool execution failed.",
        )


class TestToolPort:
    def test_tool_port_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            ToolPort()  # type: ignore[abstract]

    def test_fake_tool_is_valid_tool_port(self) -> None:
        tool = FakeTool()

        assert isinstance(tool, ToolPort)

    @pytest.mark.asyncio
    async def test_execute_accepts_tool_request(self) -> None:
        tool = FakeTool()

        request = ToolRequest(
            tool_name="search",
            input={
                "query": "AI",
            },
        )

        result = await tool.execute(request)

        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_receives_original_request(self) -> None:
        tool = FakeTool()

        request = ToolRequest(
            tool_name="browser",
            input={
                "url": "https://example.com",
            },
            metadata={
                "request_id": "req-001",
            },
        )

        await tool.execute(request)

        assert tool.received_request is request

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self) -> None:
        tool = FakeTool()

        request = ToolRequest(
            tool_name="search",
            input="query",
        )

        result = await tool.execute(request)

        assert result.output == {
            "message": "Tool executed successfully.",
        }

    @pytest.mark.asyncio
    async def test_failed_execution_uses_tool_result_contract(
        self,
    ) -> None:
        tool = FailingTool()

        request = ToolRequest(
            tool_name="database",
            input={
                "query": "SELECT 1",
            },
        )

        result = await tool.execute(request)

        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.FAILED
        assert result.error == "Tool execution failed."

    @pytest.mark.asyncio
    async def test_tool_port_is_provider_independent(self) -> None:
        tool = FakeTool()

        request = ToolRequest(
            tool_name="external_api",
            input={
                "provider": "test-provider",
                "operation": "test-operation",
            },
        )

        result = await tool.execute(request)

        assert result.status == ToolStatus.COMPLETED
        assert tool.received_request is request