from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.video.providers.local.serialization import (
    SerializedFrame,
)


@dataclass(frozen=True, slots=True)
class VisionRuntimeRequest:
    """
    Provider-independent request passed to a local vision runtime.

    This contract represents an inference request after application-level
    vision intent has been transformed into runtime-ready data.

    It intentionally contains no Ollama-specific request objects,
    endpoint names, SDK types, or response structures.
    """

    model: str
    system_prompt: str
    user_prompt: str
    frames: tuple[SerializedFrame, ...] = ()
    temperature: float = 0.1
    keep_alive: str | None = "5m"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model = self.model.strip()
        system_prompt = self.system_prompt.strip()
        user_prompt = self.user_prompt.strip()

        if not model:
            raise ValueError(
                "Vision runtime model must not be empty."
            )

        if not system_prompt:
            raise ValueError(
                "Vision runtime system prompt must not be empty."
            )

        if not user_prompt:
            raise ValueError(
                "Vision runtime user prompt must not be empty."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "Vision runtime temperature must be between 0.0 and 2.0."
            )

        if self.keep_alive is not None:
            keep_alive = self.keep_alive.strip()

            if not keep_alive:
                raise ValueError(
                    "Vision runtime keep-alive must not be empty "
                    "when provided."
                )

            object.__setattr__(
                self,
                "keep_alive",
                keep_alive,
            )

        object.__setattr__(
            self,
            "model",
            model,
        )

        object.__setattr__(
            self,
            "system_prompt",
            system_prompt,
        )

        object.__setattr__(
            self,
            "user_prompt",
            user_prompt,
        )

        object.__setattr__(
            self,
            "frames",
            tuple(self.frames),
        )


@dataclass(frozen=True, slots=True)
class VisionRuntimeResponse:
    """
    Provider-independent response returned by a local vision runtime.

    The runtime adapter is responsible for translating its native
    response into this stable contract.
    """

    text: str
    model: str
    runtime_name: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = self.text.strip()
        model = self.model.strip()
        runtime_name = self.runtime_name.strip()

        if not text:
            raise ValueError(
                "Vision runtime response text must not be empty."
            )

        if not model:
            raise ValueError(
                "Vision runtime response model must not be empty."
            )

        if not runtime_name:
            raise ValueError(
                "Vision runtime response runtime name "
                "must not be empty."
            )

        object.__setattr__(
            self,
            "text",
            text,
        )

        object.__setattr__(
            self,
            "model",
            model,
        )

        object.__setattr__(
            self,
            "runtime_name",
            runtime_name,
        )


class VisionRuntime(Protocol):
    """
    Runtime abstraction for local vision inference.

    Concrete implementations may use Ollama, vLLM, llama.cpp,
    another local runtime, or a future inference backend.
    """

    name: str

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        """
        Execute a vision inference request.
        """
        ...