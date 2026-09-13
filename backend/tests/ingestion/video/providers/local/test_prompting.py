from __future__ import annotations

import pytest

from app.ingestion.video.providers.local.prompting import (
    DefaultVisionPromptBuilder,
    VisionPrompt,
    VisionPromptBuilder,
)
from app.ingestion.video.schemas import VisionRequest, VideoFrame


def make_frame(
    *,
    frame_index: int = 1,
    timestamp_seconds: float = 2.5,
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        image=b"\xff\xd8test\xff\xd9",
    )


def make_request(
    *,
    frames: list[VideoFrame] | None = None,
    prompt: str | None = None,
    detail_level: str = "standard",
) -> VisionRequest:
    return VisionRequest(
        frames=frames if frames is not None else [make_frame()],
        prompt=prompt,
        detail_level=detail_level,
    )


class TestVisionPrompt:
    def test_combined_contains_system_and_user_prompt(self) -> None:
        prompt = VisionPrompt(
            system_prompt="system instruction",
            user_prompt="user instruction",
        )

        assert "system instruction" in prompt.combined
        assert "user instruction" in prompt.combined

    def test_combined_preserves_prompt_order(self) -> None:
        prompt = VisionPrompt(
            system_prompt="SYSTEM",
            user_prompt="USER",
        )

        assert prompt.combined == "SYSTEM\n\nUSER"

    def test_prompt_is_immutable(self) -> None:
        prompt = VisionPrompt(
            system_prompt="system",
            user_prompt="user",
        )

        with pytest.raises(AttributeError):
            prompt.system_prompt = "changed"


class TestDefaultVisionPromptBuilder:
    def test_builder_has_stable_name(self) -> None:
        builder = DefaultVisionPromptBuilder()

        assert builder.name == "default"

    def test_build_returns_vision_prompt(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert isinstance(result, VisionPrompt)

    def test_system_prompt_contains_visual_analysis_role(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert "visual analysis" in result.system_prompt.lower()

    def test_system_prompt_discourages_hallucination(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        combined = result.combined.lower()

        assert "do not invent" in combined
        assert "do not fabricate" in combined

    # -----------------------------------------------------------------------
    # Detail level
    # -----------------------------------------------------------------------

    def test_standard_detail_level_is_supported(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(detail_level="standard"),
        )

        assert "balanced detail" in result.user_prompt.lower()

    def test_low_detail_level_is_supported(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(detail_level="low"),
        )

        assert "brief description" in result.user_prompt.lower()

    def test_high_detail_level_is_supported(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(detail_level="high"),
        )

        assert "detailed visual analysis" in result.user_prompt.lower()

    def test_detail_level_is_case_insensitive(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(detail_level="HIGH"),
        )

        assert "detailed visual analysis" in result.user_prompt.lower()

    def test_unknown_detail_level_falls_back_to_standard(self) -> None:
        builder = DefaultVisionPromptBuilder()

        standard = builder.build(
            make_request(detail_level="standard"),
        )

        unknown = builder.build(
            make_request(detail_level="unsupported"),
        )

        assert unknown.user_prompt == standard.user_prompt

    # -----------------------------------------------------------------------
    # Custom prompt
    # -----------------------------------------------------------------------

    def test_custom_prompt_is_included(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(
                prompt="Focus on security-related visual indicators.",
            ),
        )

        assert (
            "Focus on security-related visual indicators."
            in result.user_prompt
        )

    def test_custom_prompt_is_trimmed(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(
                prompt="   Focus on visible text.   ",
            ),
        )

        assert "Focus on visible text." in result.user_prompt
        assert "   Focus on visible text.   " not in result.user_prompt

    def test_empty_custom_prompt_is_not_added(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(prompt="   "),
        )

        assert (
            "Additional analysis instruction:"
            not in result.user_prompt
        )

    # -----------------------------------------------------------------------
    # Frame context
    # -----------------------------------------------------------------------

    def test_frame_index_is_included(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(
                frames=[
                    make_frame(frame_index=17),
                ],
            ),
        )

        assert "frame_index=17" in result.user_prompt

    def test_timestamp_is_included(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(
                frames=[
                    make_frame(timestamp_seconds=23.75),
                ],
            ),
        )

        assert "timestamp_seconds=23.750" in result.user_prompt

    def test_multiple_frames_are_included_in_order(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(
                frames=[
                    make_frame(
                        frame_index=3,
                        timestamp_seconds=6.0,
                    ),
                    make_frame(
                        frame_index=4,
                        timestamp_seconds=8.0,
                    ),
                    make_frame(
                        frame_index=5,
                        timestamp_seconds=10.0,
                    ),
                ],
            ),
        )

        prompt = result.user_prompt

        first = prompt.index("frame_index=3")
        second = prompt.index("frame_index=4")
        third = prompt.index("frame_index=5")

        assert first < second < third

    # -----------------------------------------------------------------------
    # Structured output requirements
    # -----------------------------------------------------------------------

    def test_output_requirements_are_included(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert "objects" in result.user_prompt
        assert "entities" in result.user_prompt
        assert "actions" in result.user_prompt
        assert "visible text" in result.user_prompt

    def test_prompt_requires_valid_json(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert "ONLY valid JSON" in result.user_prompt

    def test_prompt_forbids_markdown_code_fences(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert (
            "Do not use Markdown code fences"
            in result.user_prompt
        )

    def test_prompt_forbids_extra_text_around_json(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert (
            "Do not add explanations before or after the JSON"
            in result.user_prompt
        )

    def test_prompt_requires_observations_array(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert '"observations"' in result.user_prompt

    def test_prompt_contains_structured_observation_fields(
        self,
    ) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        prompt = result.user_prompt

        assert '"timestamp_seconds"' in prompt
        assert '"frame_index"' in prompt
        assert '"description"' in prompt
        assert '"objects"' in prompt
        assert '"entities"' in prompt
        assert '"actions"' in prompt
        assert '"scene"' in prompt
        assert '"visible_text"' in prompt
        assert '"confidence"' in prompt

    def test_prompt_contains_required_structured_output_section(
        self,
    ) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert (
            "Required structured output:"
            in result.user_prompt
        )

    def test_prompt_contains_json_structure_example(
        self,
    ) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        prompt = result.user_prompt

        assert "observations" in prompt
        assert "description" in prompt
        assert "objects" in prompt
        assert "entities" in prompt
        assert "actions" in prompt
        assert "scene" in prompt
        assert "visible_text" in prompt
        assert "confidence" in prompt

    def test_prompt_requires_factual_structured_output(
        self,
    ) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        combined = result.combined.lower()

        assert "visually supported" in combined
        assert "supported by the supplied frame" in combined

    def test_prompt_requests_visible_text_analysis(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert "visible text" in result.combined.lower()

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def test_none_request_is_rejected(self) -> None:
        builder = DefaultVisionPromptBuilder()

        with pytest.raises(
            ValueError,
            match="Vision request must be provided",
        ):
            builder.build(None)  # type: ignore[arg-type]

    def test_empty_frames_are_rejected(self) -> None:
        builder = DefaultVisionPromptBuilder()

        request = make_request(frames=[])

        with pytest.raises(
            ValueError,
            match="at least one frame",
        ):
            builder.build(request)

    # -----------------------------------------------------------------------
    # Immutability / side effects
    # -----------------------------------------------------------------------

    def test_builder_does_not_modify_request(self) -> None:
        builder = DefaultVisionPromptBuilder()

        frames = [
            make_frame(
                frame_index=7,
                timestamp_seconds=14.0,
            ),
        ]

        request = make_request(
            frames=frames,
            prompt="Analyze the scene.",
            detail_level="high",
        )

        original_prompt = request.prompt
        original_detail_level = request.detail_level
        original_frames = list(request.frames)

        builder.build(request)

        assert request.prompt == original_prompt
        assert request.detail_level == original_detail_level
        assert request.frames == original_frames

    # -----------------------------------------------------------------------
    # Protocol
    # -----------------------------------------------------------------------

    def test_protocol_can_be_implemented(self) -> None:
        builder: VisionPromptBuilder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        assert isinstance(result, VisionPrompt)

    # -----------------------------------------------------------------------
    # Determinism
    # -----------------------------------------------------------------------

    def test_prompt_generation_is_deterministic(self) -> None:
        builder = DefaultVisionPromptBuilder()

        request = make_request(
            frames=[
                make_frame(
                    frame_index=2,
                    timestamp_seconds=4.0,
                ),
            ],
            prompt="Analyze the frame.",
            detail_level="high",
        )

        first = builder.build(request)
        second = builder.build(request)

        assert first == second

    # -----------------------------------------------------------------------
    # Runtime independence
    # -----------------------------------------------------------------------

    def test_prompt_is_runtime_independent(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        combined = result.combined.lower()

        assert "ollama" not in combined
        assert "openai" not in combined
        assert "gemini" not in combined
        assert "vllm" not in combined

    # -----------------------------------------------------------------------
    # Factual analysis
    # -----------------------------------------------------------------------

    def test_prompt_requests_factual_analysis(self) -> None:
        builder = DefaultVisionPromptBuilder()

        result = builder.build(
            make_request(),
        )

        combined = result.combined.lower()

        assert "factually" in combined
        assert "visually supported" in combined