from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
# Test runtime
# ============================================================================


@dataclass
class MetadataRuntime:
    """
    Deterministic fake runtime used to verify metadata propagation.

    This runtime never performs real inference.
    """

    response_text: str
    metadata: dict[str, Any]

    name: str = "test-runtime"
    model: str = "test-vision-model"

    async def generate(
        self,
        request: VisionRuntimeRequest,
    ) -> VisionRuntimeResponse:
        return VisionRuntimeResponse(
            text=self.response_text,
            model=self.model,
            runtime_name=self.name,
            metadata=dict(self.metadata),
        )


# ============================================================================
# Helpers
# ============================================================================


def make_frame(
    *,
    frame_index: int = 7,
    timestamp_seconds: float = 12.5,
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        image=b"\xff\xd8fake-image\xff\xd9",
        width=1280,
        height=720,
    )


def make_request(
    *,
    frame_index: int = 7,
    timestamp_seconds: float = 12.5,
) -> VisionRequest:
    return VisionRequest(
        frames=[
            make_frame(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
        ],
    )


def make_success_runtime() -> MetadataRuntime:
    return MetadataRuntime(
        response_text="""
        {
            "observations": [
                {
                    "frame_index": 999,
                    "timestamp_seconds": 9999.0,
                    "description": "A person standing outdoors.",
                    "objects": ["person"],
                    "scene": "outdoor",
                    "confidence": 0.92
                }
            ]
        }
        """,
        metadata={
            "total_duration": 123456,
            "load_duration": 1234,
            "prompt_eval_count": 42,
            "eval_count": 18,
            "custom_runtime_marker": "metadata-test",
        },
    )


def make_provider(
    runtime: MetadataRuntime,
) -> LocalVisionProvider:
    return LocalVisionProvider(
        runtime,
    )


def make_service(
    runtime: MetadataRuntime,
) -> VisionService:
    provider = make_provider(runtime)

    return VisionService(
        provider,
    )


# ============================================================================
# Runtime metadata propagation
# ============================================================================


async def test_runtime_metadata_is_preserved_in_provider_result() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.COMPLETED

    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"

    runtime_metadata = result.metadata[
        "runtime_metadata"
    ]

    assert runtime_metadata["total_duration"] == 123456
    assert runtime_metadata["load_duration"] == 1234
    assert runtime_metadata["prompt_eval_count"] == 42
    assert runtime_metadata["eval_count"] == 18
    assert runtime_metadata["custom_runtime_marker"] == (
        "metadata-test"
    )


# ============================================================================
# Parser metadata propagation
# ============================================================================


async def test_parser_metadata_is_preserved_in_provider_result() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.COMPLETED

    assert result.metadata["parser"] == "default"
    assert result.metadata["source_format"] == "json"


# ============================================================================
# Provider metadata
# ============================================================================


async def test_provider_metadata_identifies_execution_provider() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert result.metadata["provider"] == "local"
    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"


# ============================================================================
# Frame provenance
# ============================================================================


async def test_source_frame_index_is_authoritative() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(
            frame_index=7,
            timestamp_seconds=12.5,
        ),
    )

    assert result.status == VisionStatus.COMPLETED
    assert len(result.observations) == 1

    observation = result.observations[0]

    # Runtime deliberately returned frame_index=999.
    assert observation.frame_index == 7


async def test_source_frame_timestamp_is_authoritative() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(
            frame_index=7,
            timestamp_seconds=12.5,
        ),
    )

    assert result.status == VisionStatus.COMPLETED
    assert len(result.observations) == 1

    observation = result.observations[0]

    # Runtime deliberately returned timestamp_seconds=9999.0.
    assert observation.timestamp_seconds == 12.5


async def test_model_cannot_overwrite_frame_provenance() -> None:
    runtime = MetadataRuntime(
        response_text="""
        {
            "observations": [
                {
                    "frame_index": 100000,
                    "timestamp_seconds": 999999.0,
                    "description": "Detected scene."
                }
            ]
        }
        """,
        metadata={
            "marker": "provenance-test",
        },
    )

    provider = make_provider(runtime)

    request = make_request(
        frame_index=42,
        timestamp_seconds=18.75,
    )

    result = await provider.analyze(
        request,
    )

    assert result.status == VisionStatus.COMPLETED

    observation = result.observations[0]

    assert observation.frame_index == 42
    assert observation.timestamp_seconds == 18.75


# ============================================================================
# Observation metadata
# ============================================================================


async def test_observation_contains_parser_metadata() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    observation = result.observations[0]

    assert observation.metadata["parser"] == "default"
    assert observation.metadata["source_format"] == "json"


# ============================================================================
# Service metadata propagation
# ============================================================================


async def test_service_preserves_provider_metadata() -> None:
    runtime = make_success_runtime()

    service = make_service(runtime)

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.COMPLETED

    # Service-owned top-level metadata.
    assert result.metadata["service"] == "vision"
    assert result.metadata["provider"] == "local"

    # Provider-owned metadata remains intact.
    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"
    assert result.metadata["parser"] == "default"
    assert result.metadata["source_format"] == "json"


async def test_service_adds_service_execution_metadata() -> None:
    runtime = make_success_runtime()

    service = make_service(runtime)

    result = await service.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.COMPLETED

    assert "service_execution" in result.metadata

    service_execution = result.metadata[
        "service_execution"
    ]

    assert isinstance(
        service_execution,
        dict,
    )

    assert service_execution["requested_frames"] == 1
    assert service_execution["processed_frames"] == 1
    assert service_execution["failed_frames"] == 0

    assert isinstance(
        service_execution["duration_ms"],
        float,
    )

    assert service_execution["duration_ms"] >= 0.0


async def test_service_metadata_does_not_remove_provider_metadata() -> None:
    runtime = make_success_runtime()

    service = make_service(runtime)

    result = await service.analyze(
        make_request(),
    )

    assert result.metadata["service"] == "vision"
    assert result.metadata["provider"] == "local"

    # service_execution is a separate namespace.
    service_execution = result.metadata[
        "service_execution"
    ]

    assert "requested_frames" in service_execution
    assert "processed_frames" in service_execution
    assert "failed_frames" in service_execution
    assert "duration_ms" in service_execution

    # Provider metadata is still available at the top level.
    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"
    assert result.metadata["parser"] == "default"
    assert result.metadata["source_format"] == "json"


async def test_service_metadata_does_not_remove_runtime_metadata() -> None:
    runtime = make_success_runtime()

    service = make_service(runtime)

    result = await service.analyze(
        make_request(),
    )

    runtime_metadata = result.metadata[
        "runtime_metadata"
    ]

    assert runtime_metadata["custom_runtime_marker"] == (
        "metadata-test"
    )


# ============================================================================
# Metadata collision protection
# ============================================================================


async def test_service_execution_metadata_is_service_owned_namespace() -> None:
    runtime = make_success_runtime()

    service = make_service(runtime)

    result = await service.analyze(
        make_request(),
    )

    service_execution = result.metadata[
        "service_execution"
    ]

    assert service_execution["requested_frames"] == 1
    assert service_execution["processed_frames"] == 1
    assert service_execution["failed_frames"] == 0

    # Provider identity remains top-level and is not moved
    # or overwritten by the service_execution namespace.
    assert result.metadata["provider"] == "local"
    assert result.metadata["service"] == "vision"


async def test_runtime_metadata_remains_nested_under_runtime_metadata() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert "runtime_metadata" in result.metadata

    runtime_metadata = result.metadata[
        "runtime_metadata"
    ]

    assert isinstance(
        runtime_metadata,
        dict,
    )

    assert runtime_metadata["total_duration"] == 123456


# ============================================================================
# Complete metadata chain
# ============================================================================


async def test_complete_metadata_chain_is_preserved() -> None:
    runtime = make_success_runtime()

    service = make_service(runtime)

    result = await service.analyze(
        make_request(
            frame_index=15,
            timestamp_seconds=27.25,
        ),
    )

    assert result.status == VisionStatus.COMPLETED

    # Service metadata.
    assert result.metadata["service"] == "vision"
    assert result.metadata["provider"] == "local"

    # Runtime metadata.
    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"

    # Parser metadata.
    assert result.metadata["parser"] == "default"
    assert result.metadata["source_format"] == "json"

    # Runtime-specific metadata.
    assert result.metadata["runtime_metadata"][
        "custom_runtime_marker"
    ] == "metadata-test"

    # Service execution metadata.
    service_execution = result.metadata[
        "service_execution"
    ]

    assert service_execution["requested_frames"] == 1
    assert service_execution["processed_frames"] == 1
    assert service_execution["failed_frames"] == 0
    assert service_execution["duration_ms"] >= 0.0

    # Frame provenance.
    observation = result.observations[0]

    assert observation.frame_index == 15
    assert observation.timestamp_seconds == 27.25

    # Observation-level parser provenance.
    assert observation.metadata["parser"] == "default"
    assert observation.metadata["source_format"] == "json"


# ============================================================================
# Empty-observation provenance
# ============================================================================


async def test_empty_observation_failure_contains_provider_metadata() -> None:
    runtime = MetadataRuntime(
        response_text="""
        {
            "observations": []
        }
        """,
        metadata={
            "failure-test": True,
        },
    )

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED

    assert result.metadata["provider"] == "local"
    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"

    failure = result.metadata["failure"]

    assert failure["stage"] == "response_validation"
    assert failure["exception_type"] == "ValueError"
    assert failure["code"] == "empty_observations"

    runtime_metadata = result.metadata[
        "runtime_metadata"
    ]

    assert runtime_metadata["failure-test"] is True

    parser_metadata = result.metadata[
        "parser_metadata"
    ]

    assert parser_metadata["parser"] == "default"
    assert parser_metadata["source_format"] == "json"
    assert isinstance(
        parser_metadata["warnings"],
        list,
    )


# ============================================================================
# Unstructured response provenance
# ============================================================================


async def test_unstructured_response_contains_failure_provenance() -> None:
    runtime = MetadataRuntime(
        response_text=(
            "The model returned an unstructured response."
        ),
        metadata={
            "unstructured-test": True,
        },
    )

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert result.status == VisionStatus.FAILED

    assert result.metadata["provider"] == "local"
    assert result.metadata["runtime"] == "test-runtime"
    assert result.metadata["model"] == "test-vision-model"

    failure = result.metadata["failure"]

    assert failure["stage"] == "response_validation"
    assert failure["exception_type"] == "ValueError"
    assert failure["code"] == "unstructured_response"

    runtime_metadata = result.metadata[
        "runtime_metadata"
    ]

    assert runtime_metadata["unstructured-test"] is True

    parser_metadata = result.metadata[
        "parser_metadata"
    ]

    assert parser_metadata["parser"] == "default"
    assert parser_metadata["source_format"] != "json"
    assert isinstance(
        parser_metadata["warnings"],
        list,
    )


# ============================================================================
# Result metadata contract
# ============================================================================


async def test_result_metadata_is_a_dictionary() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    assert isinstance(
        result,
        VisionResult,
    )

    assert isinstance(
        result.metadata,
        dict,
    )


async def test_observation_metadata_is_a_dictionary() -> None:
    runtime = make_success_runtime()

    provider = make_provider(runtime)

    result = await provider.analyze(
        make_request(),
    )

    observation = result.observations[0]

    assert isinstance(
        observation.metadata,
        dict,
    )