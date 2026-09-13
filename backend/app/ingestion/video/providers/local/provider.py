from __future__ import annotations

from collections.abc import Sequence

from app.ingestion.video.provider import VisionProvider
from app.ingestion.video.schemas import (
    VideoFrame,
    VisionObservation,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from app.ingestion.video.providers.local.parsing import (
    VisionResponseParser,
)
from app.ingestion.video.providers.local.prompting import (
    DefaultVisionPromptBuilder,
    VisionPromptBuilder,
)
from app.ingestion.video.providers.local.runtime import (
    VisionRuntime,
    VisionRuntimeRequest,
)


class LocalVisionProvider:
    """
    Provider implementation for local vision inference.

    The provider coordinates:

        VisionRequest
            ↓
        Prompt Builder
            ↓
        Frame Serialization
            ↓
        VisionRuntime
            ↓
        VisionResponseParser
            ↓
        VisionObservation[]
            ↓
        VisionResult

    Runtime-specific behavior remains inside VisionRuntime.
    """

    name = "local"

    def __init__(
        self,
        runtime: VisionRuntime,
        *,
        prompt_builder: VisionPromptBuilder | None = None,
        response_parser: VisionResponseParser | None = None,
    ) -> None:
        if runtime is None:
            raise ValueError(
                "Vision runtime must be provided."
            )

        self._runtime = runtime

        self._prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else DefaultVisionPromptBuilder()
        )

        self._response_parser = (
            response_parser
            if response_parser is not None
            else VisionResponseParser()
        )

    @property
    def runtime(self) -> VisionRuntime:
        return self._runtime

    @property
    def prompt_builder(self) -> VisionPromptBuilder:
        return self._prompt_builder

    @property
    def response_parser(self) -> VisionResponseParser:
        return self._response_parser

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        """
        Execute local vision analysis.

        Runtime and parsing failures are converted into structured
        VisionResult objects.

        The original frame index and timestamp supplied by the
        application remain authoritative.
        """

        if request is None:
            raise ValueError(
                "Vision request must be provided."
            )

        if not request.frames:
            return VisionResult(
                status=VisionStatus.NO_FRAMES,
                observations=[],
                requested_frames=0,
                processed_frames=0,
                failed_frames=0,
                errors=[],
                metadata={
                    "provider": self.name,
                    "runtime": self._runtime_name(),
                },
            )

        requested_frames = len(request.frames)

        # ==================================================================
        # Prompt construction
        # ==================================================================

        try:
            prompt = self._prompt_builder.build(
                request,
            )
        except Exception as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="prompt_building",
            )

        # ==================================================================
        # Frame serialization
        # ==================================================================

        try:
            serialized_frames = self._serialize_frames(
                request.frames,
            )
        except Exception as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="frame_serialization",
            )

        # ==================================================================
        # Runtime request construction
        # ==================================================================

        try:
            runtime_request = VisionRuntimeRequest(
                model=self._get_runtime_model(),
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                frames=serialized_frames,
                temperature=self._get_runtime_temperature(),
                keep_alive=self._get_runtime_keep_alive(),
                metadata={
                    "provider": self.name,
                    "frame_count": requested_frames,
                },
            )
        except Exception as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="runtime_request",
            )

        # ==================================================================
        # Runtime invocation
        # ==================================================================

        try:
            runtime_response = await self._runtime.generate(
                runtime_request,
            )

        except TimeoutError as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="runtime",
                error_code="runtime_timeout",
            )

        except ConnectionError as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="runtime",
                error_code="runtime_connection_error",
            )

        except Exception as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="runtime",
                error_code="runtime_error",
            )

        # ==================================================================
        # Response parsing
        # ==================================================================

        try:
            parsed = self._response_parser.parse(
                runtime_response,
            )

        except Exception as exc:
            return self._build_failure_result(
                request=request,
                error=exc,
                stage="response_parsing",
                error_code="response_parse_error",
                runtime_response=runtime_response,
            )

        # ==================================================================
        # Structured-output contract
        # ==================================================================

        if parsed.source_format != "json":
            return self._build_failure_result(
                request=request,
                error=ValueError(
                    "Vision runtime did not return structured JSON."
                ),
                stage="response_validation",
                error_code="unstructured_response",
                runtime_response=runtime_response,
                parser_metadata={
                    "parser": parsed.parser_name,
                    "source_format": parsed.source_format,
                    "warnings": list(parsed.warnings),
                },
            )

        if not parsed.observations:
            return self._build_failure_result(
                request=request,
                error=ValueError(
                    "Vision runtime returned no structured observations."
                ),
                stage="response_validation",
                error_code="empty_observations",
                runtime_response=runtime_response,
                parser_metadata={
                    "parser": parsed.parser_name,
                    "source_format": parsed.source_format,
                    "warnings": list(parsed.warnings),
                },
            )

        # ==================================================================
        # Restore authoritative frame metadata
        # ==================================================================

        observations = self._restore_frame_context(
            request=request,
            observations=parsed.observations,
        )

        # ==================================================================
        # Determine result status
        # ==================================================================

        status = self._determine_status(
            requested_frames=requested_frames,
            observations=observations,
        )

        return VisionResult(
            status=status,
            observations=observations,
            requested_frames=requested_frames,
            processed_frames=len(observations),
            failed_frames=max(
                requested_frames - len(observations),
                0,
            ),
            errors=list(parsed.warnings),
            metadata={
                "provider": self.name,
                "runtime": runtime_response.runtime_name,
                "model": runtime_response.model,
                "parser": parsed.parser_name,
                "source_format": parsed.source_format,
                "runtime_metadata": runtime_response.metadata,
            },
        )

    # ======================================================================
    # Failure handling
    # ======================================================================

    def _build_failure_result(
        self,
        *,
        request: VisionRequest,
        error: Exception,
        stage: str,
        error_code: str | None = None,
        runtime_response: object | None = None,
        parser_metadata: dict | None = None,
    ) -> VisionResult:
        """
        Convert an internal provider failure into a structured
        VisionResult.

        Provider-level exceptions are not allowed to leak through
        analyze() after a valid request has been accepted.
        """

        requested_frames = len(request.frames)

        error_message = str(error).strip()

        if not error_message:
            error_message = error.__class__.__name__

        errors = [
            f"{stage}: {error_message}",
        ]

        metadata: dict = {
            "provider": self.name,
            "runtime": self._runtime_name(),
            "failure": {
                "stage": stage,
                "exception_type": error.__class__.__name__,
            },
        }

        if error_code is not None:
            metadata["failure"]["code"] = error_code

        if runtime_response is not None:
            runtime_name = getattr(
                runtime_response,
                "runtime_name",
                None,
            )

            model = getattr(
                runtime_response,
                "model",
                None,
            )

            response_metadata = getattr(
                runtime_response,
                "metadata",
                None,
            )

            if isinstance(runtime_name, str) and runtime_name:
                metadata["runtime"] = runtime_name

            if isinstance(model, str) and model:
                metadata["model"] = model

            if isinstance(response_metadata, dict):
                metadata["runtime_metadata"] = response_metadata

        if parser_metadata is not None:
            metadata["parser_metadata"] = parser_metadata

        return VisionResult(
            status=VisionStatus.FAILED,
            observations=[],
            requested_frames=requested_frames,
            processed_frames=0,
            failed_frames=requested_frames,
            errors=errors,
            metadata=metadata,
        )

    # ======================================================================
    # Runtime configuration
    # ======================================================================

    def _runtime_name(self) -> str:
        """
        Safely obtain the configured runtime name.
        """

        runtime_name = getattr(
            self._runtime,
            "name",
            None,
        )

        if isinstance(runtime_name, str) and runtime_name.strip():
            return runtime_name.strip()

        return "unknown"

    def _get_runtime_model(self) -> str:
        """
        Obtain the configured runtime model.
        """

        config = getattr(
            self._runtime,
            "config",
            None,
        )

        model = getattr(
            config,
            "model",
            None,
        )

        if isinstance(model, str) and model.strip():
            return model.strip()

        return "unknown"

    def _get_runtime_temperature(self) -> float:
        """
        Obtain the configured runtime temperature.
        """

        config = getattr(
            self._runtime,
            "config",
            None,
        )

        temperature = getattr(
            config,
            "temperature",
            0.1,
        )

        if isinstance(
            temperature,
            (int, float),
        ):
            return float(temperature)

        return 0.1

    def _get_runtime_keep_alive(
        self,
    ) -> str | None:
        """
        Obtain the configured runtime keep-alive value.
        """

        config = getattr(
            self._runtime,
            "config",
            None,
        )

        keep_alive = getattr(
            config,
            "keep_alive",
            "5m",
        )

        if keep_alive is None:
            return None

        if isinstance(
            keep_alive,
            str,
        ):
            return keep_alive.strip() or None

        return "5m"

    # ======================================================================
    # Frame serialization
    # ======================================================================

    @staticmethod
    def _serialize_frames(
        frames: Sequence[VideoFrame],
    ) -> tuple:
        """
        Serialize VideoFrame objects at the local-provider boundary.
        """

        from app.ingestion.video.providers.local.serialization import (
            Base64FrameSerializer,
        )

        serializer = Base64FrameSerializer()

        return tuple(
            serializer.serialize(frame)
            for frame in frames
        )

    # ======================================================================
    # Frame metadata preservation
    # ======================================================================

    @staticmethod
    def _restore_frame_context(
        *,
        request: VisionRequest,
        observations: Sequence[VisionObservation],
    ) -> list[VisionObservation]:
        """
        Restore authoritative frame index and timestamp information.

        Model-generated timestamps and frame indexes are not
        authoritative. The original application frame metadata is.
        """

        request_frames = {
            frame.frame_index: frame
            for frame in request.frames
        }

        restored: list[VisionObservation] = []

        for observation in observations:
            source_frame = request_frames.get(
                observation.frame_index,
            )

            if source_frame is None:
                source_frame = request.frames[0]

            restored.append(
                observation.model_copy(
                    update={
                        "frame_index": source_frame.frame_index,
                        "timestamp_seconds": (
                            source_frame.timestamp_seconds
                        ),
                    },
                )
            )

        return restored

    # ======================================================================
    # Result status
    # ======================================================================

    @staticmethod
    def _determine_status(
        *,
        requested_frames: int,
        observations: Sequence[VisionObservation],
    ) -> VisionStatus:
        """
        Determine whether all requested frames produced observations.
        """

        if not observations:
            return VisionStatus.FAILED

        if len(observations) >= requested_frames:
            return VisionStatus.COMPLETED

        return VisionStatus.PARTIAL


def ensure_vision_provider(
    provider: object,
) -> VisionProvider:
    """
    Validate that an object implements the VisionProvider contract.
    """

    analyze = getattr(
        provider,
        "analyze",
        None,
    )

    if not callable(analyze):
        raise TypeError(
            "Configured object does not implement the "
            "VisionProvider contract."
        )

    return provider  # type: ignore[return-value]