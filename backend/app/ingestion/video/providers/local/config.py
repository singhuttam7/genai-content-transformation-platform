from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalVisionConfig:
    """
    Configuration for the local vision provider.

    This configuration describes how a local vision provider should
    execute inference. It does not expose any Ollama-specific API
    objects or implementation details.

    The configuration can therefore be reused by different local
    vision runtimes in the future.
    """

    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    max_retries: int = 1
    temperature: float = 0.1
    keep_alive: str | None = "5m"

    def __post_init__(self) -> None:
        model = self.model.strip()
        base_url = self.base_url.strip()

        if not model:
            raise ValueError("Vision model must not be empty.")

        if not base_url:
            raise ValueError("Vision runtime base URL must not be empty.")

        if self.timeout_seconds <= 0:
            raise ValueError(
                "Vision runtime timeout must be greater than zero."
            )

        if self.max_retries < 0:
            raise ValueError(
                "Vision runtime max retries must be zero or greater."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "Vision temperature must be between 0.0 and 2.0."
            )

        if self.keep_alive is not None and not self.keep_alive.strip():
            raise ValueError(
                "Vision runtime keep-alive must not be empty when provided."
            )

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))

        if self.keep_alive is not None:
            object.__setattr__(
                self,
                "keep_alive",
                self.keep_alive.strip(),
            )