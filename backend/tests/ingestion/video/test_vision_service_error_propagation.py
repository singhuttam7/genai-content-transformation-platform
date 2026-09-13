from __future__ import annotations

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
    VisionResult,
    VisionStatus,
)
from app.ingestion.video.vision import VisionService


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
    Create a valid VisionRequest.
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
# Runtime implementations
# ============================================================================


class TimeoutRuntime:
    """
    Runtime that simulates a vision inference timeout.
    """

    name = "timeout-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        raise TimeoutError(
            "Vision runtime timed out."
        )


class ConnectionFailureRuntime:
    """
    Runtime that simulates a runtime connection failure.
    """

    name = "connection-failure-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        raise ConnectionError(
            "Unable to connect to vision runtime."
        )


class GenericFailureRuntime:
    """
    Runtime that simulates an unexpected runtime failure.
    """

    name = "generic-failure-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        raise RuntimeError(
            "Unexpected vision runtime failure."
        )


class UnstructuredResponseRuntime:
    """
    Runtime that returns plain text instead of the required
    structured JSON response.
    """

    name = "unstructured-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        return VisionRuntimeResponse(
            text="This is not structured JSON.",
            model="test-model",
            runtime_name=self.name,
            metadata={},
        )


class EmptyObservationRuntime:
    """
    Runtime that returns valid JSON but no observations.
    """

    name = "empty-observation-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        return VisionRuntimeResponse(
            text='{"observations": []}',
            model="test-model",
            runtime_name=self.name,
            metadata={},
        )


class SuccessfulRuntime:
    """
    Runtime that returns one valid structured observation.
    """

    name = "successful-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        return VisionRuntimeResponse(
            text=(
                '{"observations": ['
                "{"
                '"frame_index": 0,'
                '"timestamp_seconds": 999.0,'
                '"description": "Successful observation."'
                "}"
                "]}"
            ),
            model="test-model",
            runtime_name=self.name,
            metadata={
                "runtime_test": True,
            },
        )


class PartialRuntime:
    """
    Runtime that returns fewer observations than requested frames.
    """

    name = "partial-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        return VisionRuntimeResponse(
            text=(
                '{"observations": ['
                "{"
                '"frame_index": 0,'
                '"timestamp_seconds": 999.0,'
                '"description": "Partial observation."'
                "}"
                "]}"
            ),
            model="test-model",
            runtime_name=self.name,
            metadata={},
        )


# ============================================================================
# Provider factory
# ============================================================================


def make_service(runtime: object) -> VisionService:
    """
    Build VisionService with the LocalVisionProvider.
    """

    provider = LocalVisionProvider(
        runtime,  # type: ignore[arg-type]
    )

    return VisionService(
        provider,
    )


# ============================================================================
# Timeout propagation
# ============================================================================


@pytest.mark.asyncio
async def test_timeout_becomes_structured_failed_result():
    service = make_service(
        TimeoutRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert isinstance(
        result,
        VisionResult,
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_timeout_does_not_escape_application_boundary():
    service = make_service(
        TimeoutRuntime(),
    )

    result = await service.analyze(
        make_request(
            frame_count=3,
        ),
    )

    assert result.status == VisionStatus.FAILED
    assert result.requested_frames == 3
    assert result.processed_frames == 0
    assert result.failed_frames == 3


@pytest.mark.asyncio
async def test_timeout_contains_provider_failure_metadata():
    service = make_service(
        TimeoutRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.metadata["provider"] == "local"
    assert result.metadata["runtime"] == "timeout-runtime"

    failure = result.metadata["failure"]

    assert failure["stage"] == "runtime"
    assert failure["code"] == "runtime_timeout"


# ============================================================================
# Connection failure propagation
# ============================================================================


@pytest.mark.asyncio
async def test_connection_failure_becomes_structured_failed_result():
    service = make_service(
        ConnectionFailureRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_connection_failure_preserves_frame_accounting():
    service = make_service(
        ConnectionFailureRuntime(),
    )

    result = await service.analyze(
        make_request(
            frame_count=4,
        ),
    )

    assert result.requested_frames == 4
    assert result.processed_frames == 0
    assert result.failed_frames == 4


@pytest.mark.asyncio
async def test_connection_failure_contains_correct_failure_code():
    service = make_service(
        ConnectionFailureRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    failure = result.metadata["failure"]

    assert failure["stage"] == "runtime"
    assert failure["code"] == "runtime_connection_error"


# ============================================================================
# Generic runtime failure propagation
# ============================================================================


@pytest.mark.asyncio
async def test_generic_runtime_failure_becomes_structured_failed_result():
    service = make_service(
        GenericFailureRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_generic_runtime_failure_contains_runtime_error_code():
    service = make_service(
        GenericFailureRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    failure = result.metadata["failure"]

    assert failure["stage"] == "runtime"
    assert failure["code"] == "runtime_error"


# ============================================================================
# Unstructured response propagation
# ============================================================================


@pytest.mark.asyncio
async def test_unstructured_runtime_response_becomes_failed_result():
    service = make_service(
        UnstructuredResponseRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_unstructured_response_contains_validation_failure():
    service = make_service(
        UnstructuredResponseRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    failure = result.metadata["failure"]

    assert failure["stage"] == "response_validation"
    assert failure["code"] == "unstructured_response"


@pytest.mark.asyncio
async def test_unstructured_response_preserves_runtime_metadata():
    service = make_service(
        UnstructuredResponseRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.metadata["runtime"] == (
        "unstructured-runtime"
    )


# ============================================================================
# Empty observation propagation
# ============================================================================


@pytest.mark.asyncio
async def test_empty_observations_become_failed_result():
    service = make_service(
        EmptyObservationRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_empty_observations_have_correct_failure_code():
    service = make_service(
        EmptyObservationRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    failure = result.metadata["failure"]

    assert failure["stage"] == "response_validation"
    assert failure["code"] == "empty_observations"


# ============================================================================
# Successful status propagation
# ============================================================================


@pytest.mark.asyncio
async def test_successful_local_provider_result_reaches_service():
    service = make_service(
        SuccessfulRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.COMPLETED


@pytest.mark.asyncio
async def test_successful_result_contains_observation():
    service = make_service(
        SuccessfulRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert len(result.observations) == 1
    assert (
        result.observations[0].description
        == "Successful observation."
    )


@pytest.mark.asyncio
async def test_successful_result_preserves_runtime_metadata():
    service = make_service(
        SuccessfulRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    assert (
        result.metadata["runtime_metadata"]["runtime_test"]
        is True
    )


# ============================================================================
# Partial status propagation
# ============================================================================


@pytest.mark.asyncio
async def test_partial_provider_result_reaches_service():
    service = make_service(
        PartialRuntime(),
    )

    result = await service.analyze(
        make_request(
            frame_count=3,
        ),
    )

    assert result.status == VisionStatus.PARTIAL


@pytest.mark.asyncio
async def test_partial_result_preserves_frame_accounting():
    service = make_service(
        PartialRuntime(),
    )

    result = await service.analyze(
        make_request(
            frame_count=3,
        ),
    )

    assert result.requested_frames == 3
    assert result.processed_frames == 1
    assert result.failed_frames == 2


# ============================================================================
# Authoritative frame metadata propagation
# ============================================================================


@pytest.mark.asyncio
async def test_service_preserves_authoritative_frame_index():
    service = make_service(
        SuccessfulRuntime(),
    )

    request = VisionRequest(
        frames=[
            make_frame(
                frame_index=42,
                timestamp_seconds=17.5,
            )
        ],
    )

    result = await service.analyze(
        request,
    )

    assert result.observations[0].frame_index == 42


@pytest.mark.asyncio
async def test_service_preserves_authoritative_timestamp():
    service = make_service(
        SuccessfulRuntime(),
    )

    request = VisionRequest(
        frames=[
            make_frame(
                frame_index=42,
                timestamp_seconds=17.5,
            )
        ],
    )

    result = await service.analyze(
        request,
    )

    assert (
        result.observations[0].timestamp_seconds
        == 17.5
    )


# ============================================================================
# Service execution metadata
# ============================================================================


@pytest.mark.asyncio
async def test_failure_result_contains_service_execution_metadata():
    service = make_service(
        TimeoutRuntime(),
    )

    result = await service.analyze(
        make_request(
            frame_count=2,
        ),
    )

    assert "service_execution" in result.metadata

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 2
    assert execution["processed_frames"] == 0
    assert execution["failed_frames"] == 2


@pytest.mark.asyncio
async def test_success_result_contains_service_execution_metadata():
    service = make_service(
        SuccessfulRuntime(),
    )

    result = await service.analyze(
        make_request(),
    )

    execution = result.metadata[
        "service_execution"
    ]

    assert execution["requested_frames"] == 1
    assert execution["processed_frames"] == 1
    assert execution["failed_frames"] == 0
    assert execution["duration_ms"] >= 0.0