from __future__ import annotations

import time

from app.ingestion.video.provider import VisionProvider
from app.ingestion.video.schemas import (
    VisionRequest,
    VisionResult,
    VisionStatus,
)


class VisionService:
    """
    Application-level service for visual analysis.

    The service depends on the provider-independent
    VisionProvider protocol and therefore remains decoupled
    from any specific vision model, SDK, runtime, or API.

    The service validates and prepares provider requests,
    invokes the configured provider, preserves valid provider
    results, converts provider execution failures into
    structured failure results, enforces the VisionResult
    contract, and records provider-independent execution
    metadata.
    """

    def __init__(
        self,
        provider: VisionProvider,
    ) -> None:
        """
        Initialize the vision service.

        Args:
            provider:
                Vision provider implementation responsible for
                performing the actual visual analysis.

        Raises:
            ValueError:
                If no provider is supplied.
        """

        if provider is None:
            raise ValueError(
                "Vision provider must be provided."
            )

        self._provider = provider

    @property
    def provider(self) -> VisionProvider:
        """
        Return the configured vision provider.
        """

        return self._provider

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        """
        Analyze video frames using the configured vision provider.

        The service validates the request, applies the optional
        observation limit, invokes the provider exactly once,
        preserves valid provider results, converts provider
        execution exceptions into structured failure results,
        rejects invalid provider result types, and records
        provider-independent execution metadata.

        Args:
            request:
                Provider-independent vision analysis request.

        Returns:
            Provider-independent VisionResult.

        Raises:
            ValueError:
                If the request is not provided.
        """

        if request is None:
            raise ValueError(
                "Vision request must be provided."
            )

        if not request.frames:
            return VisionResult(
                status=VisionStatus.NO_FRAMES,
                requested_frames=0,
                processed_frames=0,
                failed_frames=0,
                metadata={
                    "service": "vision",
                    "provider": self._provider.name,
                    "service_execution": {
                        "requested_frames": 0,
                        "processed_frames": 0,
                        "failed_frames": 0,
                        "duration_ms": 0.0,
                    },
                },
            )

        bounded_request = self._prepare_request(request)

        started_at = time.perf_counter()

        try:
            result = await self._provider.analyze(
                bounded_request
            )
        except Exception as exc:
            duration_ms = self._calculate_duration_ms(
                started_at
            )

            return self._build_provider_failure_result(
                request=bounded_request,
                exception=exc,
                duration_ms=duration_ms,
            )

        duration_ms = self._calculate_duration_ms(
            started_at
        )

        if not isinstance(result, VisionResult):
            return self._build_invalid_result_failure(
                request=bounded_request,
                result=result,
                duration_ms=duration_ms,
            )

        self._attach_execution_metadata(
            result=result,
            duration_ms=duration_ms,
        )

        return result

    def _build_provider_failure_result(
        self,
        *,
        request: VisionRequest,
        exception: Exception,
        duration_ms: float,
    ) -> VisionResult:
        """
        Convert a provider execution exception into a
        structured VisionResult.

        The original exception object is intentionally not
        stored in the result because VisionResult is a
        serialization-friendly boundary.

        Only safe diagnostic information is retained:
        exception type and string representation.
        """

        error_message = str(exception).strip()

        if not error_message:
            error_message = (
                "Vision provider execution failed."
            )

        return VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=len(request.frames),
            processed_frames=0,
            failed_frames=len(request.frames),
            errors=[
                error_message,
            ],
            metadata={
                "service": "vision",
                "provider": self._provider.name,
                "exception_type": type(exception).__name__,
                "service_execution": {
                    "requested_frames": len(request.frames),
                    "processed_frames": 0,
                    "failed_frames": len(request.frames),
                    "duration_ms": duration_ms,
                },
            },
        )

    def _build_invalid_result_failure(
        self,
        *,
        request: VisionRequest,
        result: object,
        duration_ms: float,
    ) -> VisionResult:
        """
        Convert an invalid provider result into a structured
        VisionResult failure.

        The actual invalid object is deliberately not included
        in the returned metadata. This keeps the result
        serialization-safe and avoids accidentally exposing
        arbitrary provider objects.

        Args:
            request:
                Prepared request sent to the provider.

            result:
                Unexpected provider return value.

            duration_ms:
                Provider execution duration.

        Returns:
            Structured failed VisionResult.
        """

        return VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=len(request.frames),
            processed_frames=0,
            failed_frames=len(request.frames),
            errors=[
                "Vision provider returned an invalid result type."
            ],
            metadata={
                "service": "vision",
                "provider": self._provider.name,
                "result_type": type(result).__name__,
                "service_execution": {
                    "requested_frames": len(request.frames),
                    "processed_frames": 0,
                    "failed_frames": len(request.frames),
                    "duration_ms": duration_ms,
                },
            },
        )

    def _attach_execution_metadata(
        self,
        *,
        result: VisionResult,
        duration_ms: float,
    ) -> None:
        """
        Attach provider-independent execution metadata to a
        valid VisionResult.

        Existing provider metadata is preserved.

        Service-owned execution information is stored under
        the ``service_execution`` metadata namespace.
        """

        existing_metadata = dict(result.metadata)

        existing_metadata["service_execution"] = {
            "requested_frames": result.requested_frames,
            "processed_frames": result.processed_frames,
            "failed_frames": result.failed_frames,
            "duration_ms": duration_ms,
        }

        if "service" not in existing_metadata:
            existing_metadata["service"] = "vision"

        if "provider" not in existing_metadata:
            existing_metadata["provider"] = (
                self._provider.name
            )

        result.metadata = existing_metadata

    @staticmethod
    def _calculate_duration_ms(
        started_at: float,
    ) -> float:
        """
        Calculate elapsed provider execution time in
        milliseconds using a monotonic clock.

        A monotonic clock is used because it is appropriate
        for measuring elapsed durations and is unaffected by
        system clock adjustments.
        """

        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        return round(
            max(duration_ms, 0.0),
            3,
        )

    @staticmethod
    def _prepare_request(
        request: VisionRequest,
    ) -> VisionRequest:
        """
        Prepare a bounded provider request without mutating
        the caller's original request.

        When max_observations is supplied, only the first
        max_observations frames are forwarded.

        Frame ordering and timestamps are preserved exactly.
        """

        if request.max_observations is None:
            return request.model_copy(
                deep=True,
            )

        bounded_frames = request.frames[
            : request.max_observations
        ]

        return request.model_copy(
            update={
                "frames": bounded_frames,
            },
            deep=True,
        )