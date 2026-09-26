from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.base import AgentStatus, AgentState
from app.orchestration.contracts import (
    WorkflowRequest,
    WorkflowResult,
    WorkflowStep,
)


def test_workflow_step_requires_agent_name() -> None:
    with pytest.raises(ValidationError):
        WorkflowStep(
            agent_name="",
            task="Do something",
        )


def test_workflow_step_requires_task() -> None:
    with pytest.raises(ValidationError):
        WorkflowStep(
            agent_name="agent",
            task="",
        )


def test_workflow_step_defaults_input_to_none() -> None:
    step = WorkflowStep(
        agent_name="agent",
        task="Do something",
    )

    assert step.input is None
    assert step.metadata == {}


def test_workflow_request_requires_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        WorkflowRequest(
            input="input",
            steps=[],
        )


def test_workflow_request_preserves_metadata() -> None:
    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="task",
            )
        ],
        metadata={
            "request_id": "123",
        },
    )

    assert request.metadata["request_id"] == "123"


def test_workflow_result_accepts_completed_result() -> None:
    result = WorkflowResult(
        status=AgentStatus.COMPLETED,
        output="result",
        state=AgentState(
            status=AgentStatus.COMPLETED,
        ),
    )

    assert result.output == "result"
    assert result.error is None


def test_failed_workflow_result_requires_error() -> None:
    with pytest.raises(
        ValueError,
        match="Failed workflow results",
    ):
        WorkflowResult(
            status=AgentStatus.FAILED,
            state=AgentState(
                status=AgentStatus.FAILED,
            ),
        )


def test_successful_workflow_result_rejects_error() -> None:
    with pytest.raises(
        ValueError,
        match="Only failed workflow results",
    ):
        WorkflowResult(
            status=AgentStatus.COMPLETED,
            state=AgentState(
                status=AgentStatus.COMPLETED,
            ),
            error="unexpected",
        )


def test_workflow_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        WorkflowStep(
            agent_name="agent",
            task="task",
            unexpected="value",
        )