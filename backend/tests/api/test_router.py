from fastapi import APIRouter
from fastapi.routing import APIRoute

from app.api.router import api_router
from app.api.v1.health import router as health_router
from app.api.v1.transformations import (
    router as transformations_router,
)


def _included_routers() -> list[APIRouter]:
    """Return routers included in the application API router."""

    routers: list[APIRouter] = []

    for route in api_router.routes:
        original_router = getattr(route, "original_router", None)

        if isinstance(original_router, APIRouter):
            routers.append(original_router)

    return routers


def _route_paths(router: APIRouter) -> set[str]:
    """Return concrete route paths from a child router."""

    return {
        route.path
        for route in router.routes
        if isinstance(route, APIRoute)
    }


def test_health_router_is_an_api_router() -> None:
    assert isinstance(health_router, APIRouter)


def test_transformation_router_is_an_api_router() -> None:
    assert isinstance(
        transformations_router,
        APIRouter,
    )


def test_health_router_prefix_is_correct() -> None:
    assert health_router.prefix == "/health"


def test_transformation_router_prefix_is_correct() -> None:
    assert transformations_router.prefix == "/transformations"


def test_health_router_is_included() -> None:
    assert health_router in _included_routers()


def test_transformation_router_is_included() -> None:
    assert transformations_router in _included_routers()


def test_transformation_router_has_post_route() -> None:
    paths = _route_paths(transformations_router)

    assert "/transformations" in paths


def test_transformation_router_has_get_by_id_route() -> None:
    paths = _route_paths(transformations_router)

    assert "/transformations/{transformation_id}" in paths