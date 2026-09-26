from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


class APIError(BaseModel):
    """
    Stable machine-readable API error payload.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: Any = None


class APIErrorResponse(BaseModel):
    """
    Standard API error envelope.
    """

    model_config = ConfigDict(extra="forbid")

    error: APIError


def _normalize_validation_errors(
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert FastAPI/Pydantic validation errors into JSON-safe details.

    Pydantic may place exception objects inside ``ctx``. Those objects
    must never be returned directly through the API response.
    """

    normalized: list[dict[str, Any]] = []

    for error in errors:
        item: dict[str, Any] = {
            "type": error.get("type"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg"),
        }

        ctx = error.get("ctx")

        if isinstance(ctx, dict):
            safe_ctx: dict[str, Any] = {}

            for key, value in ctx.items():
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    safe_ctx[key] = value
                else:
                    safe_ctx[key] = str(value)

            if safe_ctx:
                item["ctx"] = safe_ctx

        normalized.append(item)

    return normalized


def _normalize_http_detail(
    detail: Any,
) -> Any:
    """
    Normalize an HTTPException detail into JSON-safe data.
    """

    if isinstance(detail, BaseModel):
        return detail.model_dump()

    if isinstance(
        detail,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return detail

    if isinstance(detail, dict):
        return {
            str(key): _normalize_http_detail(value)
            for key, value in detail.items()
        }

    if isinstance(detail, (list, tuple)):
        return [
            _normalize_http_detail(value)
            for value in detail
        ]

    return str(detail)


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """
    Convert FastAPI HTTP exceptions into the standard API envelope.
    """

    if isinstance(exc.detail, str):
        message = exc.detail
        details = None
    else:
        message = "Request could not be completed."
        details = _normalize_http_detail(
            exc.detail,
        )

    payload = APIErrorResponse(
        error=APIError(
            code=f"HTTP_{exc.status_code}",
            message=message,
            details=details,
        )
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(),
        headers=exc.headers,
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Convert request validation failures into the standard API envelope.
    """

    payload = APIErrorResponse(
        error=APIError(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=_normalize_validation_errors(
                exc.errors(),
            ),
        )
    )

    return JSONResponse(
        status_code=422,
        content=payload.model_dump(),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Return a safe generic response for unexpected application errors.

    Internal exception messages are deliberately not exposed.
    """

    payload = APIErrorResponse(
        error=APIError(
            code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred.",
        )
    )

    return JSONResponse(
        status_code=500,
        content=payload.model_dump(),
    )