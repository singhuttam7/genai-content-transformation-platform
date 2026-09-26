from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WorkflowFailurePolicy(BaseModel):
    """
    Policy controlling workflow-level recovery behavior.

    This policy is intentionally independent of concrete agents,
    LLM providers, tools, and orchestration implementations.
    """

    model_config = ConfigDict(extra="forbid")

    retry_failed_agents: bool = False

    max_attempts: int = Field(
        default=1,
        ge=1,
    )

    def __init__(self, **data: object) -> None:
        super().__init__(**data)

        if not self.retry_failed_agents and self.max_attempts != 1:
            raise ValueError(
                "max_attempts must be 1 when retry_failed_agents is False."
            )