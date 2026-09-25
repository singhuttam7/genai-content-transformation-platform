from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


LLMMessageRole = Literal[
    "system",
    "user",
    "assistant",
]


class LLMModelInfo(BaseModel):
    """Provider-independent identity and capabilities of an LLM."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    provider: str = Field(
        min_length=1,
        max_length=100,
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
    )

    context_window: int | None = Field(
        default=None,
        ge=1,
    )

    supports_streaming: bool = False

    supports_structured_output: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("provider", "model_name")
    @classmethod
    def reject_blank_identity(
        cls,
        value: str,
    ) -> str:
        """Reject provider/model names containing only whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Value cannot be blank."
            )

        return normalized


class LLMMessage(BaseModel):
    """Represent one message in an LLM conversation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    role: LLMMessageRole

    content: str = Field(
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("content")
    @classmethod
    def reject_blank_content(
        cls,
        value: str,
    ) -> str:
        """Reject messages containing only whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Message content cannot be blank."
            )

        return normalized


class LLMRequest(BaseModel):
    """Provider-independent request for one LLM generation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    messages: list[LLMMessage] = Field(
        min_length=1,
    )

    model: LLMModelInfo

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int | None = Field(
        default=None,
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(
        cls,
        value: list[LLMMessage],
    ) -> list[LLMMessage]:
        """Validate that the request contains usable messages."""

        if not value:
            raise ValueError(
                "At least one message is required."
            )

        return value


class LLMUsage(BaseModel):
    """Token usage reported by an LLM provider."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    completion_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
    )


class LLMResponse(BaseModel):
    """Provider-independent result of one LLM generation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    text: str = Field(
        min_length=1,
    )

    model: LLMModelInfo

    usage: LLMUsage | None = None

    finish_reason: str | None = Field(
        default=None,
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(
        cls,
        value: str,
    ) -> str:
        """Reject responses containing only whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Response text cannot be blank."
            )

        return normalized

    @field_validator("finish_reason")
    @classmethod
    def reject_blank_finish_reason(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject finish reasons containing only whitespace."""

        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Finish reason cannot be blank."
            )

        return normalized