from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.ingestion.video.providers.local.config import LocalVisionConfig


class RuntimeAvailabilityStatus(StrEnum):
    """
    Availability state of a local vision runtime and its requested model.
    """

    AVAILABLE = "available"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeAvailability:
    """
    Provider-independent description of local runtime availability.

    This contract deliberately does not expose any runtime-specific
    response format, SDK object, HTTP response, or implementation detail.
    """

    status: RuntimeAvailabilityStatus
    runtime_name: str
    model: str
    runtime_available: bool
    model_available: bool
    errors: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """
        Return True only when both the runtime and requested model
        are available.
        """

        return (
            self.status == RuntimeAvailabilityStatus.AVAILABLE
            and self.runtime_available
            and self.model_available
        )


class RuntimeAvailabilityChecker(Protocol):
    """
    Protocol for checking local runtime and model availability.

    Concrete implementations may target Ollama, vLLM, another local
    inference runtime, or a future runtime without changing consumers
    of this abstraction.
    """

    name: str

    async def check(
        self,
        config: LocalVisionConfig,
    ) -> RuntimeAvailability:
        """
        Check whether the configured runtime and model are available.
        """
        ...