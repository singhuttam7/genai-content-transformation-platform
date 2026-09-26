from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.agents.base import AgentStatus
from app.agents.base.state import AgentState, AgentStep


class TestAgentStep:
    def test_creates_step_with_generated_id(self) -> None:
        step = AgentStep(
            agent_name="summarizer",
            status=AgentStatus.COMPLETED,
        )

        assert isinstance(step.step_id, UUID)
        assert step.agent_name == "summarizer"
        assert step.status == AgentStatus.COMPLETED

    def test_creates_step_with_execution_data(self) -> None:
        started = datetime.now(timezone.utc)
        completed = started + timedelta(seconds=2)

        step = AgentStep(
            agent_name="summarizer",
            status=AgentStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
            output={
                "summary": "Generated summary.",
            },
            metadata={
                "model": "test-model",
            },
        )

        assert step.started_at == started
        assert step.completed_at == completed
        assert step.output == {
            "summary": "Generated summary.",
        }
        assert step.metadata["model"] == "test-model"

    def test_failed_step_requires_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Failed agent steps must include an error.",
        ):
            AgentStep(
                agent_name="validator",
                status=AgentStatus.FAILED,
            )

    def test_failed_step_accepts_error(self) -> None:
        step = AgentStep(
            agent_name="validator",
            status=AgentStatus.FAILED,
            error="Validation failed.",
        )

        assert step.error == "Validation failed."

    def test_non_failed_step_rejects_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Only failed agent steps may include an error.",
        ):
            AgentStep(
                agent_name="validator",
                status=AgentStatus.COMPLETED,
                error="Unexpected error.",
            )

    def test_completed_time_cannot_precede_start_time(self) -> None:
        started = datetime.now(timezone.utc)
        completed = started - timedelta(seconds=1)

        with pytest.raises(
            ValueError,
            match="completed_at cannot be earlier than started_at.",
        ):
            AgentStep(
                agent_name="summarizer",
                status=AgentStatus.COMPLETED,
                started_at=started,
                completed_at=completed,
            )

    def test_rejects_empty_agent_name(self) -> None:
        with pytest.raises(ValidationError):
            AgentStep(
                agent_name="",
                status=AgentStatus.RUNNING,
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentStep(
                agent_name="summarizer",
                status=AgentStatus.RUNNING,
                unexpected="value",
            )


class TestAgentState:
    def test_creates_default_state(self) -> None:
        state = AgentState()

        assert isinstance(state.execution_id, UUID)
        assert state.status == AgentStatus.PENDING
        assert state.current_agent is None
        assert state.history == []
        assert state.tool_history == []
        assert state.metadata == {}

    def test_execution_ids_are_unique(self) -> None:
        first = AgentState()
        second = AgentState()

        assert first.execution_id != second.execution_id

    def test_accepts_explicit_execution_id(self) -> None:
        execution_id = UUID(
            "12345678-1234-5678-1234-567812345678",
        )

        state = AgentState(
            execution_id=execution_id,
        )

        assert state.execution_id == execution_id

    def test_add_step_preserves_order(self) -> None:
        state = AgentState()

        first = AgentStep(
            agent_name="understanding",
            status=AgentStatus.COMPLETED,
        )

        second = AgentStep(
            agent_name="summarizer",
            status=AgentStatus.COMPLETED,
        )

        state.add_step(first)
        state.add_step(second)

        assert state.history == [
            first,
            second,
        ]

    def test_set_status(self) -> None:
        state = AgentState()

        state.set_status(AgentStatus.RUNNING)

        assert state.status == AgentStatus.RUNNING

    def test_supports_cancelled_status(self) -> None:
        state = AgentState()

        state.set_status(AgentStatus.CANCELLED)

        assert state.status == AgentStatus.CANCELLED

    def test_set_current_agent(self) -> None:
        state = AgentState()

        state.set_current_agent("summarizer")

        assert state.current_agent == "summarizer"

    def test_clear_current_agent(self) -> None:
        state = AgentState(
            current_agent="summarizer",
        )

        state.set_current_agent(None)

        assert state.current_agent is None

    def test_current_agent_is_normalized(self) -> None:
        state = AgentState()

        state.set_current_agent("  summarizer  ")

        assert state.current_agent == "summarizer"

    def test_empty_current_agent_is_rejected(self) -> None:
        state = AgentState()

        with pytest.raises(
            ValueError,
            match=(
                "current_agent must be a non-empty string or None."
            ),
        ):
            state.set_current_agent("   ")

    def test_add_tool_result_preserves_order(self) -> None:
        state = AgentState()

        first = {
            "tool": "search",
            "status": "completed",
        }

        second = {
            "tool": "browser",
            "status": "completed",
        }

        state.add_tool_result(first)
        state.add_tool_result(second)

        assert state.tool_history == [
            first,
            second,
        ]

    def test_state_metadata_is_preserved(self) -> None:
        state = AgentState(
            metadata={
                "request_id": "req-001",
                "source": "test",
            },
        )

        assert state.metadata == {
            "request_id": "req-001",
            "source": "test",
        }

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentState(
                unexpected="value",
            )