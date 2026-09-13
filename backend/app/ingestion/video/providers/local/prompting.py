from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.ingestion.video.schemas import VisionRequest


@dataclass(frozen=True, slots=True)
class VisionPrompt:
    """
    Runtime-independent prompt representation.

    The prompt is divided into system and user instructions so that
    a runtime adapter can map them to its own request format later.
    """

    system_prompt: str
    user_prompt: str

    @property
    def combined(self) -> str:
        """
        Return a deterministic combined representation.

        Runtime adapters may use the separated prompts instead when
        their target runtime supports distinct system/user messages.
        """
        return (
            f"{self.system_prompt}\n\n"
            f"{self.user_prompt}"
        )


class VisionPromptBuilder(Protocol):
    """
    Protocol for constructing provider-independent vision prompts.
    """

    name: str

    def build(
        self,
        request: VisionRequest,
    ) -> VisionPrompt:
        """
        Build a structured prompt from a vision request.
        """
        ...


class DefaultVisionPromptBuilder:
    """
    Default prompt builder for structured visual analysis.

    The builder does not depend on any particular model, runtime,
    API, SDK, or provider.

    The generated prompt explicitly requests structured JSON so
    that the model response can later be converted into the
    provider-independent VisionObservation contract.
    """

    name = "default"

    SYSTEM_PROMPT = (
        "You are a visual analysis component in a multimodal content "
        "transformation system. Analyze the supplied video frame "
        "carefully and factually. Do not invent information that is "
        "not visually supported. Distinguish clearly between visible "
        "facts and uncertain interpretations. Do not fabricate "
        "information that cannot be reliably determined from the frame."
    )

    OUTPUT_REQUIREMENTS = (
        "Return ONLY valid JSON. "
        "Do not use Markdown code fences. "
        "Do not add explanations before or after the JSON. "
        "The JSON must contain an 'observations' array. "
        "Each observation must contain only information supported "
        "by the supplied frame. "
        "Describe the visual content in a concise and factual manner. "
        "Identify relevant objects, entities, actions, scene context, "
        "and visible text when present. "
        "If something cannot be determined reliably from the frame, "
        "do not fabricate it."
    )

    STRUCTURED_OUTPUT_SCHEMA = (
        "Use exactly this JSON structure:\n"
        "{\n"
        '  "observations": [\n'
        "    {\n"
        '      "timestamp_seconds": 0.0,\n'
        '      "frame_index": 0,\n'
        '      "description": "concise factual description",\n'
        '      "objects": ["object"],\n'
        '      "entities": ["entity"],\n'
        '      "actions": ["action"],\n'
        '      "scene": "scene description",\n'
        '      "visible_text": "text visible in frame or null",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    DETAIL_INSTRUCTIONS = {
        "low": (
            "Use a brief description and focus only on the most "
            "salient visible information."
        ),
        "standard": (
            "Provide balanced detail covering the important visible "
            "objects, entities, actions, scene context, and text."
        ),
        "high": (
            "Provide detailed visual analysis, including relevant "
            "objects, entities, actions, scene context, spatially "
            "meaningful details, and visible text."
        ),
    }

    def build(
        self,
        request: VisionRequest,
    ) -> VisionPrompt:
        """
        Build a provider-independent structured vision prompt.
        """

        if request is None:
            raise ValueError(
                "Vision request must be provided."
            )

        if not request.frames:
            raise ValueError(
                "Vision request must contain at least one frame."
            )

        detail_instruction = self._get_detail_instruction(
            request.detail_level,
        )

        user_prompt = self._build_user_prompt(
            request=request,
            detail_instruction=detail_instruction,
        )

        return VisionPrompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _build_user_prompt(
        self,
        *,
        request: VisionRequest,
        detail_instruction: str,
    ) -> str:
        """
        Build the complete user-facing analysis prompt.
        """

        sections: list[str] = []

        sections.append(
            "Task:\n"
            "Analyze the supplied video frame(s) for visual content."
        )

        custom_prompt = (
            request.prompt.strip()
            if request.prompt is not None
            else ""
        )

        if custom_prompt:
            sections.append(
                "Additional analysis instruction:\n"
                f"{custom_prompt}"
            )

        sections.append(
            "Detail level:\n"
            f"{detail_instruction}"
        )

        sections.append(
            "Output requirements:\n"
            f"{self.OUTPUT_REQUIREMENTS}"
        )

        sections.append(
            "Required structured output:\n"
            f"{self.STRUCTURED_OUTPUT_SCHEMA}"
        )

        sections.append(
            "Frame context:\n"
            f"{self._build_frame_context(request)}"
        )

        return "\n\n".join(sections)

    @staticmethod
    def _build_frame_context(
        request: VisionRequest,
    ) -> str:
        """
        Build deterministic frame metadata for the model.

        Frame timestamps and indexes are supplied explicitly so
        downstream processing can preserve temporal information.
        """

        frame_lines: list[str] = []

        for frame in request.frames:
            frame_lines.append(
                "- "
                f"frame_index={frame.frame_index}, "
                f"timestamp_seconds="
                f"{frame.timestamp_seconds:.3f}"
            )

        return "\n".join(frame_lines)

    def _get_detail_instruction(
        self,
        detail_level: str,
    ) -> str:
        """
        Resolve the requested detail level.

        Unknown detail levels safely fall back to standard detail.
        """

        normalized = detail_level.strip().lower()

        if normalized in self.DETAIL_INSTRUCTIONS:
            return self.DETAIL_INSTRUCTIONS[normalized]

        return self.DETAIL_INSTRUCTIONS["standard"]