from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base.contracts import AgentRequest


class AgentExecutionContext(BaseModel):
    """
    Runtime context supplied to an agent during orchestration.

    The context keeps agent input separate from orchestration-specific
    state while allowing optional RAG information to participate in
    agent execution.
    """

    model_config = ConfigDict(extra="forbid")

    request: AgentRequest
    rag_context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)