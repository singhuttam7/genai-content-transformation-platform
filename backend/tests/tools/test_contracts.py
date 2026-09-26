from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.tools.base import (
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class TestToolStatus:
    def test_all_lifecycle_statuses_exist(self) -> None:
        assert ToolStatus.PENDING.value == "pending"
        assert ToolStatus.RUNNING.value == "running"
        assert ToolStatus.COMPLETED.value == "completed"
        assert ToolStatus.FAILED.value == "failed"
        assert ToolStatus.CANCELLED.value == "cancelled"


class TestToolRequest:
    def test_creates_valid_request(self) -> None:
        request = ToolRequest(
            tool_name="web_search",
            input={
                "query": "latest AI research",
            },
        )

        assert request.tool_name == "web_search"
        assert request.input == {
            "query": "latest AI research",
        }
        assert request.metadata == {}

    def test_accepts_metadata(self) -> None:
        request = ToolRequest(
            tool_name="database_query",
            input={
                "query": "SELECT 1",
            },
            metadata={
                "request_id": "req-001",
                "agent": "research-agent",
            },
        )

        assert request.metadata["request_id"] == "req-001"
        assert request.metadata["agent"] == "research-agent"

    def test_rejects_empty_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolRequest(
                tool_name="",
                input="data",
            )

    def test_rejects_missing_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolRequest(
                input="data",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolRequest(
                tool_name="search",
                input="data",
                unexpected="value",
            )


class TestToolResult:
    def test_creates_completed_result(self) -> None:
        result = ToolResult(
            status=ToolStatus.COMPLETED,
            output={
                "results": [
                    "result-1",
                    "result-2",
                ],
            },
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.output == {
            "results": [
                "result-1",
                "result-2",
            ],
        }
        assert result.error is None
        assert result.metadata == {}

    def test_creates_running_result(self) -> None:
        result = ToolResult(
            status=ToolStatus.RUNNING,
        )

        assert result.status == ToolStatus.RUNNING
        assert result.output is None
        assert result.error is None

    def test_creates_cancelled_result(self) -> None:
        result = ToolResult(
            status=ToolStatus.CANCELLED,
            metadata={
                "reason": "execution_cancelled",
            },
        )

        assert result.status == ToolStatus.CANCELLED
        assert result.metadata["reason"] == "execution_cancelled"
        assert result.error is None

    def test_failed_result_requires_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Failed tool results must include an error.",
        ):
            ToolResult(
                status=ToolStatus.FAILED,
            )

    def test_failed_result_accepts_error(self) -> None:
        result = ToolResult(
            status=ToolStatus.FAILED,
            error="Tool execution failed.",
        )

        assert result.status == ToolStatus.FAILED
        assert result.error == "Tool execution failed."

    def test_non_failed_result_rejects_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Only failed tool results may include an error.",
        ):
            ToolResult(
                status=ToolStatus.COMPLETED,
                output="result",
                error="unexpected error",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolResult(
                status=ToolStatus.COMPLETED,
                unexpected="value",
            )

    def test_accepts_arbitrary_output(self) -> None:
        result = ToolResult(
            status=ToolStatus.COMPLETED,
            output=[
                "item-1",
                {
                    "item": 2,
                },
            ],
        )

        assert result.output == [
            "item-1",
            {
                "item": 2,
            },
        ]

    def test_accepts_metadata(self) -> None:
        result = ToolResult(
            status=ToolStatus.COMPLETED,
            output="done",
            metadata={
                "tool": "search",
                "duration_ms": 150,
            },
        )

        assert result.metadata["tool"] == "search"
        assert result.metadata["duration_ms"] == 150