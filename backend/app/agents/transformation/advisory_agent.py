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


class AdvisoryTransformationAgent(TransformationAgentPort):
    """
    Transform source content into a professional advisory.

    The agent owns advisory-specific prompt construction and transformation
    result construction. RAG retrieval and LLM execution remain delegated
    to the frozen A7 AgentIntegrationService boundary.
    """

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.ADVISORY

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
        Execute an advisory transformation.

        The generated LLM text is wrapped in an ArtifactEnvelope and
        returned through the transformation domain contract.
        """

        self._validate_request(request)

        agent_request = AgentRequest(
            task=self._build_task(request),
            input=self._build_input(request),
            metadata={
                **request.metadata,
                "transformation_type": (
                    request.transformation_type.value
                ),
                "language": request.language,
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
            artifact_type=TransformationType.ADVISORY.value,
            title=self._build_title(request),
            content=response.text,
            metadata={
                "transformation_type": (
                    TransformationType.ADVISORY.value
                ),
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
                    TransformationType.ADVISORY.value
                ),
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
            != TransformationType.ADVISORY
        ):
            raise ValueError(
                "AdvisoryTransformationAgent only accepts "
                "ADVISORY transformation requests."
            )

        if request.input is None:
            raise ValueError(
                "Advisory transformation input must not be None."
            )

        if (
            isinstance(request.input, str)
            and not request.input.strip()
        ):
            raise ValueError(
                "Advisory transformation input must not be empty."
            )

    @staticmethod
    def _build_task(
        request: TransformationRequest,
    ) -> str:
        return (
            "Transform the supplied source material into a "
            "professional advisory."
        )

    @classmethod
    def _build_input(
        cls,
        request: TransformationRequest,
    ) -> str:
        source = cls._serialize_input(request.input)

        sections = [
            "Create a professional advisory from the source "
            "material below.",
            "",
            "SOURCE MATERIAL:",
            source,
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
            sections.extend(
                [
                    "",
                    "CONFIGURATION:",
                    cls._serialize_input(
                        request.configuration,
                    ),
                ]
            )

        sections.extend(
            [
                "",
                "ADVISORY REQUIREMENTS:",
                "- Preserve factual meaning from the source.",
                "- Use clear and professional language.",
                "- Organize information for the intended audience.",
                "- Do not invent unsupported facts.",
                "- Make important actions, warnings, or "
                "recommendations easy to identify.",
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
                    "Advisory input string must not be empty."
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
    ) -> str:
        if request.objective:
            objective = request.objective.strip()

            if objective:
                return f"Advisory — {objective}"

        return "Generated Advisory"