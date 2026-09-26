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


class InfographicTransformationAgent(TransformationAgentPort):
    """
    Transform source material into a structured infographic content package.

    The agent generates the content hierarchy, key facts, sections,
    visual recommendations, and presentation guidance. Actual image
    rendering remains outside this transformation agent.
    """

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.INFOGRAPHIC

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

        format_type = self._get_format(request)

        agent_request = AgentRequest(
            task=self._build_task(
                request,
                format_type,
            ),
            input=self._build_input(
                request,
                format_type,
            ),
            metadata={
                **request.metadata,
                "transformation_type": (
                    request.transformation_type.value
                ),
                "language": request.language,
                "format": format_type,
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
            artifact_type=TransformationType.INFOGRAPHIC.value,
            title=self._build_title(
                request,
                format_type,
            ),
            content=response.text,
            metadata={
                "transformation_type": (
                    TransformationType.INFOGRAPHIC.value
                ),
                "format": format_type,
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
                    TransformationType.INFOGRAPHIC.value
                ),
                "format": format_type,
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
            != TransformationType.INFOGRAPHIC
        ):
            raise ValueError(
                "InfographicTransformationAgent only accepts "
                "INFOGRAPHIC transformation requests."
            )

        if request.input is None:
            raise ValueError(
                "Infographic input must not be None."
            )

        if (
            isinstance(request.input, str)
            and not request.input.strip()
        ):
            raise ValueError(
                "Infographic input must not be empty."
            )

    @staticmethod
    def _get_format(
        request: TransformationRequest,
    ) -> str:
        format_type = request.configuration.get(
            "format",
            "standard",
        )

        if not isinstance(format_type, str):
            raise TypeError(
                "Infographic format must be a string."
            )

        format_type = format_type.strip()

        if not format_type:
            raise ValueError(
                "Infographic format must not be empty."
            )

        return format_type

    @staticmethod
    def _build_task(
        request: TransformationRequest,
        format_type: str,
    ) -> str:
        return (
            "Transform the supplied source material into a "
            f"structured, visually coherent infographic content "
            f"package using the {format_type} format."
        )

    @classmethod
    def _build_input(
        cls,
        request: TransformationRequest,
        format_type: str,
    ) -> str:
        source = cls._serialize_input(
            request.input
        )

        sections = [
            (
                "Create a structured infographic content package "
                "from the source material below."
            ),
            "",
            "SOURCE MATERIAL:",
            source,
            "",
            "FORMAT:",
            format_type,
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

        configuration = dict(
            request.configuration
        )
        configuration.pop(
            "format",
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
                "INFOGRAPHIC REQUIREMENTS:",
                "- Preserve the factual meaning of the source.",
                "- Do not invent unsupported facts, statistics, "
                "or claims.",
                "- Identify the most important information "
                "for visual communication.",
                "- Organize the content into clear logical sections.",
                "- Highlight key facts, figures, comparisons, "
                "or processes when supported by the source.",
                "- Provide concise text suitable for visual display.",
                "- Recommend appropriate visual elements such as "
                "charts, icons, diagrams, timelines, or callouts "
                "when useful.",
                "- Keep visual recommendations grounded in "
                "the source material.",
                "- Make the information understandable without "
                "requiring unnecessary surrounding text.",
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
                    "Infographic input must not be empty."
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
        format_type: str,
    ) -> str:
        if request.objective:
            objective = request.objective.strip()

            if objective:
                return (
                    f"Infographic — "
                    f"{objective}"
                )

        return f"Generated {format_type.title()} Infographic"