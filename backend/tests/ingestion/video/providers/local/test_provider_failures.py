from __future__ import annotations

import asyncio

import pytest

from app.ingestion.video.providers.local.provider import (
    LocalVisionProvider,
)
from app.ingestion.video.providers.local.runtime import (
    VisionRuntimeRequest,
    VisionRuntimeResponse,
)
from app.ingestion.video.schemas import (
    VideoFrame,
    VisionRequest,
    VisionStatus,
)


# ============================================================================
# Fake runtimes
# ============================================================================


class FailingVisionRuntime:
    """
    Fake runtime that raises a configured exception.
    """

    name = "failing-runtime"

    def __init__(
        self,
        exception: Exception,
    ) -> None:
        self.exception = exception
        self.requests: list[VisionRuntimeRequest] = []

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        self.requests.append(request)
        raise self.exception


class MalformedVisionRuntime:
    """
    Fake runtime that returns malformed model output.
    """

    name = "malformed-runtime"

    def __init__(
        self,
        response_text: str,
    ) -> None:
        self.response_text = response_text
        self.requests: list[VisionRuntimeRequest] = []

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        self.requests.append(request)

        return VisionRuntimeResponse(
            text=self.response_text,
            model="fake-model",
            runtime_name=self.name,
            metadata={
                "test": True,
            },
        )


# ============================================================================
# Helpers
# ============================================================================


def make_frame(
    *,
    frame_index: int = 0,
    timestamp_seconds: float = 1.0,
) -> VideoFrame:
    """
    Create a minimal valid JPEG frame.
    """

    return VideoFrame(
        timestamp_seconds=timestamp_seconds,
        frame_index=frame_index,
        image=b"\xff\xd8\xff\xd9",
    )


def make_request(
    *,
    frame_count: int = 1,
) -> VisionRequest:
    """
    Create a deterministic VisionRequest.
    """

    return VisionRequest(
        frames=[
            make_frame(
                frame_index=index,
                timestamp_seconds=float(index + 1),
            )
            for index in range(frame_count)
        ],
    )


# ============================================================================
# Timeout handling
# ============================================================================


@pytest.mark.asyncio
async def test_provider_converts_timeout_error_to_failed_result():
    runtime = FailingVisionRuntime(
        TimeoutError("vision runtime timed out")
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors

    assert any(
        "timed out" in error.lower()
        or "timeout" in error.lower()
        for error in result.errors
    )


@pytest.mark.asyncio
async def test_provider_converts_asyncio_timeout_to_failed_result():
    runtime = FailingVisionRuntime(
        asyncio.TimeoutError(
            "async vision runtime timed out"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors


# ============================================================================
# Connection failures
# ============================================================================


@pytest.mark.asyncio
async def test_provider_converts_connection_error_to_failed_result():
    runtime = FailingVisionRuntime(
        ConnectionError(
            "unable to connect to vision runtime"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors

    assert any(
        "connect" in error.lower()
        for error in result.errors
    )


# ============================================================================
# Generic runtime failures
# ============================================================================


@pytest.mark.asyncio
async def test_provider_converts_generic_runtime_exception_to_failed_result():
    runtime = FailingVisionRuntime(
        RuntimeError(
            "unexpected runtime failure"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors

    assert any(
        "unexpected runtime failure" in error
        for error in result.errors
    )


# ============================================================================
# Malformed responses
# ============================================================================


@pytest.mark.asyncio
async def test_provider_converts_malformed_response_to_failed_result():
    runtime = MalformedVisionRuntime(
        "this is not valid JSON"
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1

    assert result.errors


@pytest.mark.asyncio
async def test_provider_handles_empty_structured_observation_response():
    runtime = MalformedVisionRuntime(
        '{"observations": []}'
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 1
    assert result.processed_frames == 0
    assert result.failed_frames == 1


# ============================================================================
# Failure metadata
# ============================================================================


@pytest.mark.asyncio
async def test_provider_includes_failure_metadata():
    runtime = FailingVisionRuntime(
        RuntimeError(
            "test runtime failure"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED

    assert result.metadata["provider"] == "local"

    assert (
        result.metadata["runtime"]
        == "failing-runtime"
    )


# ============================================================================
# Runtime invocation
# ============================================================================


@pytest.mark.asyncio
async def test_provider_records_runtime_invocation_before_failure():
    runtime = FailingVisionRuntime(
        RuntimeError(
            "runtime failed"
        )
    )

    provider = LocalVisionProvider(runtime)

    request = make_request()

    await provider.analyze(request)

    assert len(runtime.requests) == 1


# ============================================================================
# Multiple frames
# ============================================================================


@pytest.mark.asyncio
async def test_provider_reports_all_frames_as_failed_when_runtime_fails():
    runtime = FailingVisionRuntime(
        RuntimeError(
            "runtime failed"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request(frame_count=4)
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 4
    assert result.processed_frames == 0
    assert result.failed_frames == 4


# ============================================================================
# No raw exception leakage
# ============================================================================


@pytest.mark.asyncio
async def test_provider_does_not_leak_runtime_exception():
    runtime = FailingVisionRuntime(
        RuntimeError(
            "internal runtime failure"
        )
    )

    provider = LocalVisionProvider(runtime)

    try:
        result = await provider.analyze(
            make_request()
        )
    except Exception as exc:
        pytest.fail(
            f"Provider leaked runtime exception: {exc!r}"
        )

    assert result.status == VisionStatus.FAILED


# ============================================================================
# Error information
# ============================================================================


@pytest.mark.asyncio
async def test_provider_error_contains_exception_information():
    runtime = FailingVisionRuntime(
        ValueError(
            "invalid model response"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.errors

    combined_errors = " ".join(
        result.errors
    )

    assert (
        "invalid model response"
        in combined_errors
    )


# ============================================================================
# Failure result contract
# ============================================================================


@pytest.mark.asyncio
async def test_failure_result_contains_empty_observations():
    runtime = FailingVisionRuntime(
        RuntimeError(
            "runtime failed"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.FAILED
    assert result.observations == []


# ============================================================================
# Timeout exception identity
# ============================================================================


@pytest.mark.asyncio
async def test_timeout_failure_is_structured_without_exception_propagation():
    runtime = FailingVisionRuntime(
        TimeoutError(
            "model inference exceeded timeout"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request(frame_count=3)
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 3
    assert result.processed_frames == 0
    assert result.failed_frames == 3

    assert len(result.errors) >= 1