from __future__ import annotations

import json
from typing import Any

from app.agents.base import AgentIntegrationService, AgentRequest
from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.agents.transformation.port import TransformationAgentPort
from app.models_ai.llm import LLMModelInfo


class VideoTransformationAgent(TransformationAgentPort):
    """
    Transform source material into a structured video production package.

    The agent generates script, scenes, storyboard directions,
    narration, captions, timing guidance, and visual recommendations.
    Actual video rendering is handled outside this transformation agent.
    """

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.VIDEO

    def __init__(
        self,
        *,
        integration_service: AgentIntegrationService,
        model: LLMModelInfo,
    ) -> None:
        if not isinstance(
            integration_service,
            AgentIntegrationService,
        ):
            raise TypeError(
                "integration_service must be an "
                "AgentIntegrationService."
            )

        if not isinstance(model, LLMModelInfo):
            raise TypeError(
                "model must be an LLMModelInfo."
            )

        self._integration_service = integration_service
        self._model = model

    async def execute(
        self,
        request: TransformationRequest,
    ) -> TransformationResult:
        self._validate_request(request)

        video_type = self._get_video_type(request)

        agent_request = AgentRequest(
            task=self._build_task(request, video_type),
            input=self._build_input(request, video_type),
            metadata={
                **request.metadata,
                "transformation_type": (
                    request.transformation_type.value
                ),
                "language": request.language,
                "video_type": video_type,
            },
        )

        agent_result = await self._integration_service.execute(
            agent_request,
            model=self._model,
            use_rag=True,
        )

        response = agent_result.output

        if not hasattr(response, "text"):
            raise TypeError(
                "Agent integration returned an invalid LLM response."
            )

        artifact = ArtifactEnvelope(
            artifact_type=TransformationType.VIDEO.value,
            title=self._build_title(
                request,
                video_type,
            ),
            content=response.text,
            metadata={
                "transformation_type": (
                    TransformationType.VIDEO.value
                ),
                "video_type": video_type,
                "language": request.language,
                "objective": request.objective,
                "audience": request.audience,
                "tone": request.tone,
                "detail_level": request.detail_level,
                "style": request.style,
            },
            provenance={
                "agent": self.__class__.__name__,
                "model_provider": self._model.provider,
                "model_name": self._model.model_name,
                "rag_enabled": True,
            },
        )

        return TransformationResult(
            status=TransformationStatus.COMPLETED,
            artifacts=[artifact],
            metadata={
                "transformation_type": (
                    TransformationType.VIDEO.value
                ),
                "video_type": video_type,
                "rag_enabled": True,
                "model_provider": self._model.provider,
                "model_name": self._model.model_name,
            },
        )

    @staticmethod
    def _validate_request(
        request: TransformationRequest,
    ) -> None:
        if not isinstance(
            request,
            TransformationRequest,
        ):
            raise TypeError(
                "request must be a TransformationRequest."
            )

        if (
            request.transformation_type
            != TransformationType.VIDEO
        ):
            raise ValueError(
                "VideoTransformationAgent only accepts "
                "VIDEO transformation requests."
            )

        if request.input is None:
            raise ValueError(
                "Video input must not be None."
            )

        if (
            isinstance(request.input, str)
            and not request.input.strip()
        ):
            raise ValueError(
                "Video input must not be empty."
            )

    @staticmethod
    def _get_video_type(
        request: TransformationRequest,
    ) -> str:
        video_type = request.configuration.get(
            "video_type",
            "explainer",
        )

        if not isinstance(video_type, str):
            raise TypeError(
                "Video type must be a string."
            )

        video_type = video_type.strip()

        if not video_type:
            raise ValueError(
                "Video type must not be empty."
            )

        return video_type

    @staticmethod
    def _build_task(
        request: TransformationRequest,
        video_type: str,
    ) -> str:
        return (
            "Transform the supplied source material into a "
            "complete, structured video production package "
            f"using a {video_type} video format."
        )

    @classmethod
    def _build_input(
        cls,
        request: TransformationRequest,
        video_type: str,
    ) -> str:
        source = cls._serialize_input(request.input)

        sections = [
            (
                "Create a complete video production package "
                "from the source material below."
            ),
            "",
            "SOURCE MATERIAL:",
            source,
            "",
            "VIDEO TYPE:",
            video_type,
        ]

        optional_fields = [
            ("OBJECTIVE", request.objective),
            ("AUDIENCE", request.audience),
            ("TONE", request.tone),
            ("LANGUAGE", request.language),
            ("DETAIL LEVEL", request.detail_level),
            ("STYLE", request.style),
        ]

        for label, value in optional_fields:
            if value is not None:
                normalized = value.strip()

                if normalized:
                    sections.extend(
                        [
                            "",
                            f"{label}:",
                            normalized,
                        ]
                    )

        configuration = dict(request.configuration)
        configuration.pop("video_type", None)

        if configuration:
            sections.extend(
                [
                    "",
                    "CONFIGURATION:",
                    cls._serialize_input(configuration),
                ]
            )

        sections.extend(
            [
                "",
                "VIDEO PRODUCTION REQUIREMENTS:",
                "- Preserve the factual meaning of the source.",
                "- Do not invent unsupported facts, statistics, "
                "claims, or quotations.",
                "- Create a clear narrative beginning, middle, "
                "and conclusion.",
                "- Divide the video into logical scenes.",
                "- Provide a concise script for every scene.",
                "- Provide narration text for spoken content.",
                "- Provide subtitle or caption text.",
                "- Provide visual directions for every scene.",
                "- Recommend suitable images, footage, graphics, "
                "charts, diagrams, or animations when useful.",
                "- Keep visual recommendations grounded in the "
                "source material.",
                "- Include approximate scene duration or timing "
                "guidance when useful.",
                "- Recommend transitions between scenes when "
                "appropriate.",
                "- Maintain consistency in terminology, tone, "
                "and narrative style.",
                "- Ensure the final scene provides an appropriate "
                "conclusion or call to action when supported "
                "by the objective.",
                "- Structure the response so it can later be "
                "consumed by a video rendering pipeline.",
            ]
        )

        return "\n".join(sections)

    @staticmethod
    def _serialize_input(
        value: Any,
    ) -> str:
        if isinstance(value, str):
            normalized = value.strip()

            if not normalized:
                raise ValueError(
                    "Video input must not be empty."
                )

            return normalized

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )

    @staticmethod
    def _build_title(
        request: TransformationRequest,
        video_type: str,
    ) -> str:
        if request.objective:
            objective = request.objective.strip()

            if objective:
                return (
                    f"Video — {objective}"
                )

        return (
            f"Generated "
            f"{video_type.title()} Video Package"
        )