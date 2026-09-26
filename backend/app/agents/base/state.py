from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base.contracts import AgentStatus


class AgentStep(BaseModel):
    """
    Immutable record describing one agent execution step.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: UUID = Field(
        default_factory=uuid4,
    )
    agent_name: str = Field(
        min_length=1,
    )
    status: AgentStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError(
                "completed_at cannot be earlier than started_at."
            )

        if self.status == AgentStatus.FAILED:
            if not self.error:
                raise ValueError(
                    "Failed agent steps must include an error."
                )
        elif self.error is not None:
            raise ValueError(
                "Only failed agent steps may include an error."
            )


class AgentState(BaseModel):
    """
    Provider-independent state for an agent workflow.

    This is the domain representation of execution state.
    Orchestration frameworks such as LangGraph may adapt to this
    representation but are not part of the domain contract.
    """

    model_config = ConfigDict(extra="forbid")

    execution_id: UUID = Field(
        default_factory=uuid4,
    )
    status: AgentStatus = AgentStatus.PENDING
    current_agent: str | None = None
    history: list[AgentStep] = Field(
        default_factory=list,
    )
    tool_history: list[dict[str, Any]] = Field(
        default_factory=list,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    def add_step(
        self,
        step: AgentStep,
    ) -> None:
        self.history.append(step)

    def set_status(
        self,
        status: AgentStatus,
    ) -> None:
        self.status = status

    def set_current_agent(
        self,
        agent_name: str | None,
    ) -> None:
        if agent_name is not None and not agent_name.strip():
            raise ValueError(
                "current_agent must be a non-empty string or None."
            )

        self.current_agent = (
            agent_name.strip()
            if agent_name is not None
            else None
        )

    def add_tool_result(
        self,
        result: dict[str, Any],
    ) -> None:
        self.tool_history.append(result)