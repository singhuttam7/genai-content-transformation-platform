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


class PresentationTransformationAgent(TransformationAgentPort):
    """
    Transform source material into a structured presentation package.

    The agent generates presentation structure, slide content,
    speaker notes, and visual recommendations. Actual presentation
    file rendering/export remains outside this transformation agent.
    """

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.PRESENTATION

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

        presentation_type = self._get_presentation_type(request)

        agent_request = AgentRequest(
            task=self._build_task(
                request,
                presentation_type,
            ),
            input=self._build_input(
                request,
                presentation_type,
            ),
            metadata={
                **request.metadata,
                "transformation_type": (
                    request.transformation_type.value
                ),
                "language": request.language,
                "presentation_type": presentation_type,
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
            artifact_type=TransformationType.PRESENTATION.value,
            title=self._build_title(
                request,
                presentation_type,
            ),
            content=response.text,
            metadata={
                "transformation_type": (
                    TransformationType.PRESENTATION.value
                ),
                "presentation_type": presentation_type,
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
                    TransformationType.PRESENTATION.value
                ),
                "presentation_type": presentation_type,
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
            != TransformationType.PRESENTATION
        ):
            raise ValueError(
                "PresentationTransformationAgent only accepts "
                "PRESENTATION transformation requests."
            )

        if request.input is None:
            raise ValueError(
                "Presentation input must not be None."
            )

        if (
            isinstance(request.input, str)
            and not request.input.strip()
        ):
            raise ValueError(
                "Presentation input must not be empty."
            )

    @staticmethod
    def _get_presentation_type(
        request: TransformationRequest,
    ) -> str:
        presentation_type = request.configuration.get(
            "presentation_type",
            "professional",
        )

        if not isinstance(
            presentation_type,
            str,
        ):
            raise TypeError(
                "Presentation type must be a string."
            )

        presentation_type = presentation_type.strip()

        if not presentation_type:
            raise ValueError(
                "Presentation type must not be empty."
            )

        return presentation_type

    @staticmethod
    def _build_task(
        request: TransformationRequest,
        presentation_type: str,
    ) -> str:
        return (
            "Transform the supplied source material into a "
            "complete, structured presentation package using "
            f"a {presentation_type} presentation style."
        )

    @classmethod
    def _build_input(
        cls,
        request: TransformationRequest,
        presentation_type: str,
    ) -> str:
        source = cls._serialize_input(
            request.input
        )

        sections = [
            (
                "Create a complete presentation package "
                "from the source material below."
            ),
            "",
            "SOURCE MATERIAL:",
            source,
            "",
            "PRESENTATION TYPE:",
            presentation_type,
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
            "presentation_type",
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
                "PRESENTATION REQUIREMENTS:",
                "- Preserve the factual meaning of the source.",
                "- Do not invent unsupported facts, statistics, "
                "claims, or quotations.",
                "- Create a logical narrative flow from beginning "
                "to conclusion.",
                "- Organize the presentation into clearly defined "
                "slides.",
                "- Give every slide a concise and meaningful title.",
                "- Keep slide content concise and presentation-ready.",
                "- Include important facts, findings, implications, "
                "decisions, risks, or actions when supported "
                "by the source.",
                "- Provide speaker notes when additional explanation "
                "would help the presenter.",
                "- Recommend appropriate visual elements such as "
                "charts, diagrams, images, timelines, or tables "
                "when useful.",
                "- Keep visual recommendations grounded in the "
                "source material.",
                "- Avoid unnecessary text density on slides.",
                "- End with an appropriate conclusion or call to action "
                "when supported by the objective.",
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
                    "Presentation input must not be empty."
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
        presentation_type: str,
    ) -> str:
        if request.objective:
            objective = request.objective.strip()

            if objective:
                return (
                    f"Presentation — "
                    f"{objective}"
                )

        return (
            f"Generated "
            f"{presentation_type.title()} Presentation"
        )