from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import AgentState, AgentStatus


class WorkflowStep(BaseModel):
    """
    One deterministic step in a workflow.

    Each step identifies the agent that must execute and the task it
    should perform. If input is omitted, the workflow input is used.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(min_length=1)
    task: str = Field(min_length=1)
    input: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    """
    Application-level request describing a workflow execution.
    """

    model_config = ConfigDict(extra="forbid")

    input: Any
    steps: list[WorkflowStep] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    """
    Final result of a workflow execution.
    """

    model_config = ConfigDict(extra="forbid")

    status: AgentStatus
    output: Any = None
    state: AgentState
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if self.status == AgentStatus.FAILED:
            if not self.error:
                raise ValueError(
                    "Failed workflow results must include an error."
                )
        elif self.error is not None:
            raise ValueError(
                "Only failed workflow results may include an error."
            )