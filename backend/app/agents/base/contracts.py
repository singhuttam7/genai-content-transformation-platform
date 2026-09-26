from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStatus(StrEnum):
    """Lifecycle status of an agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRequest(BaseModel):
    """
    Provider-independent request submitted to an agent.

    The request contains the task description and the input payload.
    Agent implementations may interpret the payload according to
    their domain while the contract remains generic.
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description="Description of the task the agent must execute.",
    )
    input: Any = Field(
        description="Input payload supplied to the agent.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution metadata.",
    )


class AgentResult(BaseModel):
    """
    Provider-independent result produced by an agent execution.
    """

    model_config = ConfigDict(extra="forbid")

    status: AgentStatus
    output: Any = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata produced by the agent.",
    )
    error: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if self.status == AgentStatus.FAILED:
            if not self.error:
                raise ValueError(
                    "Failed agent results must include an error."
                )
        elif self.error is not None:
            raise ValueError(
                "Only failed agent results may include an error."
            )