from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMProviderConfig(BaseModel):
    """
    Provider-independent configuration for an LLM adapter.

    Secrets such as API keys are represented here but must be supplied
    through the application's secure configuration layer rather than
    hard-coded in source code.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    provider: str = Field(
        min_length=1,
        max_length=100,
    )

    api_key: str | None = Field(
        default=None,
        min_length=1,
    )

    base_url: str | None = Field(
        default=None,
        min_length=1,
    )

    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
    )

    default_model: str | None = Field(
        default=None,
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "provider",
        "api_key",
        "base_url",
        "default_model",
    )
    @classmethod
    def normalize_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject blank strings and normalize surrounding whitespace."""

        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Value cannot be blank."
            )

        return normalized