from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """
    Configuration for deterministic structure-aware knowledge chunking.

    The default configuration uses:
        max_characters = 1200
        overlap_characters = 150

    When a caller specifies a smaller max_characters without explicitly
    specifying overlap_characters, the default overlap is automatically
    reduced to a safe value.

    Explicit invalid overlap values are rejected.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_characters: int = Field(
        default=1200,
        ge=1,
        description=(
            "Maximum target size of a generated chunk in characters."
        ),
    )

    overlap_characters: int = Field(
        default=150,
        ge=0,
        description=(
            "Maximum amount of text carried from the previous "
            "chunk into the next chunk."
        ),
    )

    min_characters: int = Field(
        default=1,
        ge=1,
        description="Minimum meaningful chunk size.",
    )

    preserve_headings: bool = Field(
        default=True,
        description=(
            "Whether structural heading context should be preserved "
            "and propagated into generated chunks."
        ),
    )

    preserve_page_boundaries: bool = Field(
        default=True,
        description=(
            "Whether page provenance should be preserved when "
            "available."
        ),
    )

    preserve_time_boundaries: bool = Field(
        default=True,
        description=(
            "Whether media timestamp provenance should be preserved."
        ),
    )

    include_non_text_blocks: bool = Field(
        default=False,
        description=(
            "Whether blocks without textual content should be "
            "emitted as standalone chunk candidates."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_defaults(
        cls,
        values: Any,
    ) -> Any:
        """
        Safely adapt the default overlap to a custom max size.

        Examples:

            ChunkingConfig()
                -> 1200 / 150

            ChunkingConfig(max_characters=100)
                -> 100 / 99

            ChunkingConfig(
                max_characters=100,
                overlap_characters=25,
            )
                -> 100 / 25

        An explicitly supplied overlap is never silently changed.
        """

        if not isinstance(values, dict):
            return values

        values = dict(values)

        max_characters = values.get("max_characters", 1200)

        if (
            "overlap_characters" not in values
            and isinstance(max_characters, int)
            and max_characters >= 1
        ):
            values["overlap_characters"] = min(
                150,
                max(0, max_characters - 1),
            )

        return values

    @model_validator(mode="after")
    def validate_configuration(self) -> "ChunkingConfig":
        """
        Validate relationships between chunking parameters.
        """

        if self.overlap_characters >= self.max_characters:
            raise ValueError(
                "overlap_characters must be smaller than max_characters."
            )

        if self.min_characters > self.max_characters:
            raise ValueError(
                "min_characters cannot exceed max_characters."
            )

        return self