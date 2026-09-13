from __future__ import annotations

from app.ingestion.video.schemas import (
    FrameExtractionRequest,
)


class FrameSamplingPolicy:
    """
    Deterministic policy for calculating video frame timestamps.

    This class contains no FFmpeg, filesystem, image-processing,
    or model-specific logic.

    Its only responsibility is deciding WHICH timestamps should
    be extracted.
    """

    def generate_timestamps(
        self,
        *,
        duration_seconds: float,
        request: FrameExtractionRequest,
    ) -> list[float]:
        """
        Generate timestamps according to the requested sampling
        interval and time boundaries.

        Returned timestamps are always:
        - non-negative,
        - ordered,
        - within the video duration,
        - within the requested time window,
        - limited by max_frames.
        """

        if duration_seconds < 0:
            raise ValueError(
                "duration_seconds must be non-negative."
            )

        start_time = request.start_time_seconds

        # ---------------------------------------------------------
        # No valid extraction window
        # ---------------------------------------------------------

        if start_time >= duration_seconds:
            return []

        end_time = request.end_time_seconds

        if end_time is None:
            end_time = duration_seconds
        else:
            end_time = min(
                end_time,
                duration_seconds,
            )

        if end_time < start_time:
            return []

        # ---------------------------------------------------------
        # Generate timestamps
        # ---------------------------------------------------------

        timestamps: list[float] = []

        current_time = start_time
        interval = request.interval_seconds

        epsilon = 1e-9

        while (
            current_time <= end_time + epsilon
            and len(timestamps) < request.max_frames
        ):
            # Clamp tiny floating-point overflow at the end
            # boundary.
            timestamp = min(
                current_time,
                end_time,
            )

            timestamps.append(
                timestamp
            )

            current_time += interval

        return timestamps