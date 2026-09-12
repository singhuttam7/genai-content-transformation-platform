from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.video.schemas import VideoInfo


class VideoASRCapability(StrEnum):
    """
    Describes whether a video contains an audio stream
    that can proceed to the video-to-ASR pipeline.
    """

    AUDIO_AVAILABLE = "audio_available"
    NO_AUDIO = "no_audio"


class VideoASRCapabilityResult(BaseModel):
    """
    Result of checking whether a video has an audio stream.
    """

    model_config = ConfigDict(extra="forbid")

    capability: VideoASRCapability

    reason: str

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class VideoASRCapabilityChecker:
    """
    Determines whether an inspected video contains an
    audio stream suitable for the video-to-ASR workflow.

    This component does not inspect media itself, extract
    audio, or invoke an ASR provider. It operates exclusively
    on the VideoInfo produced by the video inspection layer.
    """

    def check(
        self,
        video_info: VideoInfo,
    ) -> VideoASRCapabilityResult:
        """
        Determine whether the inspected video contains audio.
        """

        if video_info.audio is None:
            return VideoASRCapabilityResult(
                capability=VideoASRCapability.NO_AUDIO,
                reason="The video does not contain an audio stream.",
                metadata={
                    "audio_present": False,
                },
            )

        return VideoASRCapabilityResult(
            capability=VideoASRCapability.AUDIO_AVAILABLE,
            reason="The video contains an audio stream.",
            metadata={
                "audio_present": True,
                "codec_name": video_info.audio.codec_name,
                "sample_rate": video_info.audio.sample_rate,
                "channels": video_info.audio.channels,
            },
        )