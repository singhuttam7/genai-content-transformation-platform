from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.base.contracts import (
    AgentRequest,
    AgentResult,
    AgentStatus,
)


class TestAgentStatus:
    def test_all_lifecycle_statuses_exist(self) -> None:
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.CANCELLED.value == "cancelled"


class TestAgentRequest:
    def test_creates_valid_request(self) -> None:
        request = AgentRequest(
            task="summarize document",
            input={"text": "Example document."},
        )

        assert request.task == "summarize document"
        assert request.input == {
            "text": "Example document.",
        }
        assert request.metadata == {}

    def test_accepts_metadata(self) -> None:
        request = AgentRequest(
            task="transform content",
            input="source text",
            metadata={
                "request_id": "req-001",
                "source": "test",
            },
        )

        assert request.metadata["request_id"] == "req-001"
        assert request.metadata["source"] == "test"

    def test_rejects_empty_task(self) -> None:
        with pytest.raises(ValidationError):
            AgentRequest(
                task="",
                input="source",
            )

    def test_rejects_missing_task(self) -> None:
        with pytest.raises(ValidationError):
            AgentRequest(
                input="source",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentRequest(
                task="test",
                input="source",
                unexpected="value",
            )


class TestAgentResult:
    def test_creates_completed_result(self) -> None:
        result = AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "summary": "Generated summary.",
            },
        )

        assert result.status == AgentStatus.COMPLETED
        assert result.output == {
            "summary": "Generated summary.",
        }
        assert result.error is None
        assert result.metadata == {}

    def test_creates_running_result(self) -> None:
        result = AgentResult(
            status=AgentStatus.RUNNING,
        )

        assert result.status == AgentStatus.RUNNING
        assert result.output is None
        assert result.error is None

    def test_creates_cancelled_result(self) -> None:
        result = AgentResult(
            status=AgentStatus.CANCELLED,
            metadata={
                "reason": "user_cancelled",
            },
        )

        assert result.status == AgentStatus.CANCELLED
        assert result.metadata["reason"] == "user_cancelled"
        assert result.error is None

    def test_failed_result_requires_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Failed agent results must include an error.",
        ):
            AgentResult(
                status=AgentStatus.FAILED,
            )

    def test_failed_result_accepts_error(self) -> None:
        result = AgentResult(
            status=AgentStatus.FAILED,
            error="Agent execution failed.",
        )

        assert result.status == AgentStatus.FAILED
        assert result.error == "Agent execution failed."

    def test_non_failed_result_rejects_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Only failed agent results may include an error.",
        ):
            AgentResult(
                status=AgentStatus.COMPLETED,
                output="result",
                error="unexpected error",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult(
                status=AgentStatus.COMPLETED,
                unexpected="value",
            )

    def test_accepts_arbitrary_output(self) -> None:
        result = AgentResult(
            status=AgentStatus.COMPLETED,
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
        result = AgentResult(
            status=AgentStatus.COMPLETED,
            output="done",
            metadata={
                "agent": "test-agent",
                "duration_ms": 100,
            },
        )

        assert result.metadata["agent"] == "test-agent"
        assert result.metadata["duration_ms"] == 100