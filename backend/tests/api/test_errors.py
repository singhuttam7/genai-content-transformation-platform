from __future__ import annotations

import pytest
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError

from app.api.errors import (
    APIError,
    APIErrorResponse,
    _normalize_http_detail,
    _normalize_validation_errors,
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)


def make_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1234),
            "scheme": "http",
        }
    )


def test_api_error_contract_rejects_extra_fields():
    with pytest.raises(ValueError):
        APIError(
            code="TEST",
            message="Test",
            unexpected=True,
        )


def test_api_error_response_has_expected_envelope():
    response = APIErrorResponse(
        error=APIError(
            code="TEST_ERROR",
            message="Test message",
        )
    )

    assert response.model_dump() == {
        "error": {
            "code": "TEST_ERROR",
            "message": "Test message",
            "details": None,
        }
    }


def test_normalize_http_detail_handles_nested_values():
    detail = {
        "message": "invalid",
        "values": [
            1,
            True,
            None,
            {"nested": "value"},
        ],
    }

    assert _normalize_http_detail(detail) == detail


def test_normalize_http_detail_stringifies_unknown_values():
    class CustomValue:
        def __str__(self):
            return "custom-value"

    assert (
        _normalize_http_detail(CustomValue())
        == "custom-value"
    )


def test_normalize_http_detail_handles_pydantic_model():
    detail = APIError(
        code="INNER_ERROR",
        message="Inner message",
    )

    assert _normalize_http_detail(detail) == {
        "code": "INNER_ERROR",
        "message": "Inner message",
        "details": None,
    }


def test_normalize_validation_errors_sanitizes_context():
    errors = [
        {
            "type": "value_error",
            "loc": ("body", "name"),
            "msg": "Invalid value",
            "ctx": {
                "error": ValueError("bad value"),
                "limit": 10,
                "flag": True,
            },
        }
    ]

    normalized = _normalize_validation_errors(
        errors
    )

    assert normalized == [
        {
            "type": "value_error",
            "loc": ["body", "name"],
            "msg": "Invalid value",
            "ctx": {
                "error": "bad value",
                "limit": 10,
                "flag": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_http_exception_handler_uses_string_detail():
    response = await http_exception_handler(
        make_request(),
        HTTPException(
            status_code=404,
            detail="Resource not found.",
        ),
    )

    assert response.status_code == 404

    assert response.body == (
        b'{"error":{"code":"HTTP_404",'
        b'"message":"Resource not found.",'
        b'"details":null}}'
    )


@pytest.mark.asyncio
async def test_http_exception_handler_normalizes_structured_detail():
    response = await http_exception_handler(
        make_request(),
        HTTPException(
            status_code=422,
            detail={
                "field": "name",
                "reason": "invalid",
            },
        ),
    )

    assert response.status_code == 422

    assert response.body == (
        b'{"error":{"code":"HTTP_422",'
        b'"message":"Request could not be completed.",'
        b'"details":{"field":"name","reason":"invalid"}}}'
    )


@pytest.mark.asyncio
async def test_http_exception_handler_preserves_headers():
    response = await http_exception_handler(
        make_request(),
        HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ),
    )

    assert response.status_code == 401
    assert (
        response.headers["www-authenticate"]
        == "Bearer"
    )


@pytest.mark.asyncio
async def test_request_validation_exception_handler_returns_422():
    exception = RequestValidationError(
        [
            {
                "type": "missing",
                "loc": ("body", "name"),
                "msg": "Field required",
                "input": {},
            }
        ]
    )

    response = await request_validation_exception_handler(
        make_request(),
        exception,
    )

    assert response.status_code == 422

    assert response.body == (
        b'{"error":{"code":"VALIDATION_ERROR",'
        b'"message":"Request validation failed.",'
        b'"details":[{"type":"missing",'
        b'"loc":["body","name"],'
        b'"msg":"Field required"}]}}'
    )


@pytest.mark.asyncio
async def test_unhandled_exception_handler_hides_internal_error():
    response = await unhandled_exception_handler(
        make_request(),
        RuntimeError("database password leaked"),
    )

    assert response.status_code == 500

    assert response.body == (
        b'{"error":{"code":"INTERNAL_SERVER_ERROR",'
        b'"message":"An internal server error occurred.",'
        b'"details":null}}'
    )

    assert (
        b"database password leaked"
        not in response.body
    )