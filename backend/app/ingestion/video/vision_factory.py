from __future__ import annotations

from app.core.config import settings
from app.ingestion.video.provider import VisionProvider
from app.ingestion.video.providers.local.config import (
    LocalVisionConfig,
)
from app.ingestion.video.providers.local.ollama import (
    OllamaVisionRuntime,
)
from app.ingestion.video.providers.local.provider import (
    LocalVisionProvider,
)


def create_vision_provider() -> VisionProvider:
    """
    Create the configured vision provider.

    The factory is responsible for translating application-level
    settings into the concrete provider and runtime configuration.

    Supported providers:
        - local

    Raises:
        ValueError:
            If vision is disabled or an unsupported provider is
            configured.
    """

    if not settings.vision_enabled:
        raise ValueError(
            "Vision is disabled."
        )

    provider_name = (
        settings.vision_provider
        .strip()
        .lower()
    )

    if provider_name == "local":
        config = LocalVisionConfig(
            model=settings.vision_model,
            base_url=settings.vision_base_url,
            timeout_seconds=settings.vision_timeout_seconds,
            max_retries=settings.vision_max_retries,
            temperature=settings.vision_temperature,
            keep_alive=settings.vision_keep_alive,
        )

        runtime = OllamaVisionRuntime(
            config,
        )

        return LocalVisionProvider(
            runtime,
        )

    raise ValueError(
        f"Unsupported vision provider: "
        f"{settings.vision_provider}"
    )