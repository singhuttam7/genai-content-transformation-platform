from __future__ import annotations

from functools import lru_cache

from app.ingestion.video.vision import VisionService
from app.ingestion.video.vision_factory import (
    create_vision_provider,
)


@lru_cache
def get_vision_service() -> VisionService:
    """
    Return the application-wide VisionService.

    The configured vision provider is created through the
    vision factory and injected into VisionService.

    The service instance is cached so that the configured
    provider and its runtime are reused across the application.
    """

    provider = create_vision_provider()

    return VisionService(provider)