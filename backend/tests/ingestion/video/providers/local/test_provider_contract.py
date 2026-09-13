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


# ============================================================================
# Fake runtime
# ============================================================================


class ContractTestRuntime:
    """
    Minimal deterministic runtime used exclusively for contract tests.
    """

    name = "contract-test-runtime"

    def __init__(
        self,
        response_text: str = (
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 0.0,'
            '"description": "Contract test observation."'
            "}"
            "]}"
        ),
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
            model="contract-test-model",
            runtime_name=self.name,
            metadata={
                "contract_test": True,
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
# Provider identity
# ============================================================================


def test_local_provider_exposes_provider_name():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    assert provider.name == "local"


# ============================================================================
# VisionProvider contract
# ============================================================================


def test_local_provider_exposes_analyze_method():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    analyze = getattr(
        provider,
        "analyze",
        None,
    )

    assert callable(analyze)



# ============================================================================
# Successful result contract
# ============================================================================


@pytest.mark.asyncio
async def test_analyze_returns_vision_result():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert isinstance(
        result,
        VisionResult,
    )


@pytest.mark.asyncio
async def test_successful_result_has_completed_status():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.COMPLETED


@pytest.mark.asyncio
async def test_successful_result_contains_observations():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.observations
    assert len(result.observations) == 1


# ============================================================================
# Result accounting contract
# ============================================================================


@pytest.mark.asyncio
async def test_result_frame_accounting_is_consistent():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request(frame_count=1)
    )

    assert result.requested_frames == 1
    assert result.processed_frames == 1
    assert result.failed_frames == 0


@pytest.mark.asyncio
async def test_multi_frame_request_preserves_requested_count():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request(frame_count=3)
    )

    assert result.requested_frames == 3


# ============================================================================
# No-frame contract
# ============================================================================


@pytest.mark.asyncio
async def test_empty_request_returns_no_frames_status():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request(frame_count=0)
    )

    assert isinstance(
        result,
        VisionResult,
    )

    assert result.status == VisionStatus.NO_FRAMES


@pytest.mark.asyncio
async def test_empty_request_does_not_call_runtime():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    await provider.analyze(
        make_request(frame_count=0)
    )

    assert runtime.requests == []


# ============================================================================
# Provider/runtime separation
# ============================================================================


@pytest.mark.asyncio
async def test_provider_does_not_require_ollama_specific_runtime():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.status == VisionStatus.COMPLETED

    assert result.metadata["runtime"] == (
        "contract-test-runtime"
    )


@pytest.mark.asyncio
async def test_provider_passes_runtime_request_to_generic_runtime():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    await provider.analyze(
        make_request()
    )

    assert len(runtime.requests) == 1

    runtime_request = runtime.requests[0]

    assert runtime_request.system_prompt
    assert runtime_request.user_prompt
    assert runtime_request.frames


# ============================================================================
# Observation contract
# ============================================================================


@pytest.mark.asyncio
async def test_observation_preserves_frame_index():
    runtime = ContractTestRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 999.0,'
            '"description": "Frame observation."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        VisionRequest(
            frames=[
                make_frame(
                    frame_index=5,
                    timestamp_seconds=10.5,
                )
            ],
        )
    )

    assert result.observations[0].frame_index == 5


@pytest.mark.asyncio
async def test_observation_preserves_frame_timestamp():
    runtime = ContractTestRuntime(
        response_text=(
            '{"observations": ['
            "{"
            '"frame_index": 0,'
            '"timestamp_seconds": 999.0,'
            '"description": "Frame observation."'
            "}"
            "]}"
        )
    )

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        VisionRequest(
            frames=[
                make_frame(
                    frame_index=0,
                    timestamp_seconds=27.25,
                )
            ],
        )
    )

    assert (
        result.observations[0].timestamp_seconds
        == 27.25
    )


# ============================================================================
# Metadata contract
# ============================================================================


@pytest.mark.asyncio
async def test_result_contains_provider_metadata():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.metadata["provider"] == "local"


@pytest.mark.asyncio
async def test_result_contains_runtime_metadata():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert result.metadata["runtime"] == (
        "contract-test-runtime"
    )

    assert result.metadata["model"] == (
        "contract-test-model"
    )


@pytest.mark.asyncio
async def test_runtime_metadata_is_preserved():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert (
        result.metadata["runtime_metadata"]["contract_test"]
        is True
    )


# ============================================================================
# Contract result isolation
# ============================================================================


@pytest.mark.asyncio
async def test_result_is_independent_of_runtime_response_object():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert isinstance(
        result,
        VisionResult,
    )

    assert result.observations


# ============================================================================
# Failure contract
# ============================================================================


class FailingContractRuntime:
    """
    Runtime used to verify that the provider converts runtime
    failures into the VisionResult contract.
    """

    name = "failing-contract-runtime"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        raise RuntimeError(
            "contract runtime failure"
        )


@pytest.mark.asyncio
async def test_runtime_failure_still_returns_vision_result():
    runtime = FailingContractRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    assert isinstance(
        result,
        VisionResult,
    )

    assert result.status == VisionStatus.FAILED


@pytest.mark.asyncio
async def test_runtime_failure_preserves_frame_accounting():
    runtime = FailingContractRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request(frame_count=3)
    )

    assert result.requested_frames == 3
    assert result.processed_frames == 0
    assert result.failed_frames == 3


# ============================================================================
# Contract stability
# ============================================================================


@pytest.mark.asyncio
async def test_provider_result_contains_required_contract_attributes():
    runtime = ContractTestRuntime()

    provider = LocalVisionProvider(runtime)

    result = await provider.analyze(
        make_request()
    )

    required_attributes = (
        "status",
        "observations",
        "requested_frames",
        "processed_frames",
        "failed_frames",
        "errors",
        "metadata",
    )

    for attribute in required_attributes:
        assert hasattr(
            result,
            attribute,
        )