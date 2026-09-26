from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(StrEnum):
    """Lifecycle status of a tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolRequest(BaseModel):
    """
    Provider-independent request submitted to a tool.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(
        min_length=1,
        description="Canonical name of the requested tool.",
    )
    input: Any = Field(
        description="Input payload supplied to the tool.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution metadata.",
    )


class ToolResult(BaseModel):
    """
    Provider-independent result produced by a tool execution.
    """

    model_config = ConfigDict(extra="forbid")

    status: ToolStatus
    output: Any = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata produced by the tool.",
    )
    error: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if self.status == ToolStatus.FAILED:
            if not self.error:
                raise ValueError(
                    "Failed tool results must include an error."
                )
        elif self.error is not None:
            raise ValueError(
                "Only failed tool results may include an error."
            )