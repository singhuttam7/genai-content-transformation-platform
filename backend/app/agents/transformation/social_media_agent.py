from __future__ import annotations

import json
from typing import Any

from app.agents.base import (
    AgentIntegrationService,
    AgentRequest,
)
from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.agents.transformation.port import TransformationAgentPort
from app.models_ai.llm import LLMModelInfo


class SocialMediaTransformationAgent(TransformationAgentPort):
    """
    Transform source material into platform-optimized social media content.

    The agent owns social-media-specific prompt construction and result
    construction. RAG retrieval and LLM execution remain delegated to the
    frozen A7 AgentIntegrationService boundary.
    """

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.SOCIAL_MEDIA

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
        """
        Execute a social-media transformation.
        """

        self._validate_request(request)

        platform = self._get_platform(request)

        agent_request = AgentRequest(
            task=self._build_task(
                request,
                platform,
            ),
            input=self._build_input(
                request,
                platform,
            ),
            metadata={
                **request.metadata,
                "transformation_type": (
                    request.transformation_type.value
                ),
                "language": request.language,
                "platform": platform,
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
            artifact_type=TransformationType.SOCIAL_MEDIA.value,
            title=self._build_title(
                request,
                platform,
            ),
            content=response.text,
            metadata={
                "transformation_type": (
                    TransformationType.SOCIAL_MEDIA.value
                ),
                "platform": platform,
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
                    TransformationType.SOCIAL_MEDIA.value
                ),
                "platform": platform,
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
            != TransformationType.SOCIAL_MEDIA
        ):
            raise ValueError(
                "SocialMediaTransformationAgent only accepts "
                "SOCIAL_MEDIA transformation requests."
            )

        if request.input is None:
            raise ValueError(
                "Social media input must not be None."
            )

        if (
            isinstance(request.input, str)
            and not request.input.strip()
        ):
            raise ValueError(
                "Social media input must not be empty."
            )

    @staticmethod
    def _get_platform(
        request: TransformationRequest,
    ) -> str:
        platform = request.configuration.get(
            "platform",
            "LinkedIn",
        )

        if not isinstance(platform, str):
            raise TypeError(
                "Social media platform must be a string."
            )

        platform = platform.strip()

        if not platform:
            raise ValueError(
                "Social media platform must not be empty."
            )

        return platform

    @staticmethod
    def _build_task(
        request: TransformationRequest,
        platform: str,
    ) -> str:
        return (
            "Transform the supplied source material into "
            f"platform-optimized social media content for {platform}."
        )

    @classmethod
    def _build_input(
        cls,
        request: TransformationRequest,
        platform: str,
    ) -> str:
        source = cls._serialize_input(
            request.input
        )

        sections = [
            (
                "Create platform-optimized social media content "
                f"for {platform} from the source material below."
            ),
            "",
            "SOURCE MATERIAL:",
            source,
            "",
            "PLATFORM:",
            platform,
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

        if request.configuration:
            configuration = dict(
                request.configuration
            )
            configuration.pop(
                "platform",
                None,
            )

            if configuration:
                sections.extend(
                    [
                        "",
                        "CONFIGURATION:",
                        cls._serialize_input(
                            configuration
                        ),
                    ]
                )

        sections.extend(
            [
                "",
                "SOCIAL MEDIA REQUIREMENTS:",
                "- Preserve the factual meaning of the source.",
                "- Do not invent unsupported facts or claims.",
                "- Optimize the content for the specified platform.",
                "- Use clear, engaging, professional language.",
                "- Make the core message immediately understandable.",
                "- Respect the requested audience, tone, language, "
                "and style.",
                "- Use platform-appropriate structure and formatting.",
                "- Include a clear call to action when appropriate.",
                "- Use hashtags only when appropriate for the platform.",
                "- Do not add unsupported statistics, quotations, "
                "or claims.",
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
                    "Social media input must not be empty."
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
        platform: str,
    ) -> str:
        if request.objective:
            objective = request.objective.strip()

            if objective:
                return (
                    f"{platform} Post — "
                    f"{objective}"
                )

        return f"Generated {platform} Post"