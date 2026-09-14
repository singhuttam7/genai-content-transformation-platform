from __future__ import annotations

from app.api.dependencies import (
    VisionServiceDependency,
)
from app.ingestion.video.vision import VisionService
from app.ingestion.video.vision_dependencies import (
    get_vision_service,
)


def test_vision_service_dependency_annotation() -> None:
    """
    Verify that the public API dependency annotation contains
    FastAPI dependency metadata.
    """

    dependency = VisionServiceDependency

    assert dependency.__metadata__

    dependency_callable = dependency.__metadata__[0]

    assert dependency_callable.dependency is get_vision_service


def test_vision_service_dependency_uses_get_vision_service() -> None:
    """
    Verify that the FastAPI dependency points to the configured
    VisionService dependency factory.
    """

    dependency = VisionServiceDependency

    depends_object = dependency.__metadata__[0]

    assert (
        depends_object.dependency
        is get_vision_service
    )


def test_vision_service_dependency_type() -> None:
    """
    Verify that the dependency exposes VisionService as its
    annotated service type.
    """

    dependency = VisionServiceDependency

    assert dependency.__origin__ is VisionService