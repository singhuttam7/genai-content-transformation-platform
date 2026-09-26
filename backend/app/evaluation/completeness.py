from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationResult,
    TransformationType,
)
from app.evaluation.contracts import (
    EvaluationCheck,
    EvaluationCheckType,
    EvaluationIssue,
    EvaluationRequest,
    EvaluationResult,
    EvaluationSeverity,
    EvaluationStatus,
)
from app.evaluation.port import EvaluationPort


class CompletenessEvaluator(EvaluationPort):
    """Validate that transformation outputs contain required content."""

    @property
    def check_type(self) -> EvaluationCheckType:
        return EvaluationCheckType.COMPLETENESS

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest.")

        transformation_result = request.transformation_result

        if not isinstance(transformation_result, TransformationResult):
            raise TypeError(
                "transformation_result must be a TransformationResult."
            )

        issues: list[EvaluationIssue] = []

        if not transformation_result.artifacts:
            issues.append(
                self._issue(
                    code="NO_ARTIFACTS",
                    message="The transformation result contains no artifacts.",
                )
            )

        expected = self._required_components(
            request.transformation_request.transformation_type
        )

        for index, artifact in enumerate(transformation_result.artifacts):
            issues.extend(
                self._validate_artifact(
                    artifact=artifact,
                    index=index,
                    required_components=expected,
                )
            )

        status = (
            EvaluationStatus.FAILED
            if issues
            else EvaluationStatus.PASSED
        )

        score = 0.0 if issues else 1.0

        check = EvaluationCheck(
            check_type=EvaluationCheckType.COMPLETENESS,
            status=status,
            score=score,
            issues=issues,
            metadata={
                "artifact_count": len(transformation_result.artifacts),
                "required_components": expected,
            },
        )

        return EvaluationResult(
            status=status,
            score=score,
            checks=[check],
            issues=issues,
            metadata={
                "evaluator": self.__class__.__name__,
                "artifact_count": len(transformation_result.artifacts),
            },
        )

    @classmethod
    def _required_components(
        cls,
        transformation_type: TransformationType,
    ) -> tuple[str, ...]:
        requirements: dict[TransformationType, tuple[str, ...]] = {
            TransformationType.ADVISORY: (
                "title",
                "summary",
                "key_points",
                "recommendations",
            ),
            TransformationType.EXECUTIVE_SUMMARY: (
                "title",
                "summary",
                "key_findings",
                "recommendations",
            ),
            TransformationType.SOCIAL_MEDIA: (
                "platform",
                "content",
                "hashtags",
            ),
            TransformationType.INFOGRAPHIC: (
                "title",
                "sections",
                "visual_recommendations",
            ),
            TransformationType.PRESENTATION: (
                "slides",
                "speaker_notes",
                "visual_recommendations",
            ),
            TransformationType.VIDEO: (
                "script",
                "scenes",
                "narration",
                "subtitles",
                "visual_recommendations",
            ),
        }

        return requirements[transformation_type]

    @classmethod
    def _validate_artifact(
        cls,
        *,
        artifact: ArtifactEnvelope,
        index: int,
        required_components: tuple[str, ...],
    ) -> list[EvaluationIssue]:
        if not isinstance(artifact.content, Mapping):
            return [
                cls._issue(
                    code=f"ARTIFACT_{index}_STRUCTURED_CONTENT_REQUIRED",
                    message=(
                        "Completeness validation requires structured "
                        "artifact content."
                    ),
                    metadata={
                        "artifact_index": index,
                    },
                )
            ]

        issues: list[EvaluationIssue] = []

        for component in required_components:
            if component not in artifact.content:
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_MISSING_{component.upper()}",
                        message=(
                            f"Required component '{component}' "
                            "is missing from the artifact."
                        ),
                        metadata={
                            "artifact_index": index,
                            "component": component,
                        },
                    )
                )
                continue

            value = artifact.content[component]

            if cls._is_empty(value):
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_EMPTY_{component.upper()}",
                        message=(
                            f"Required component '{component}' "
                            "must not be empty."
                        ),
                        metadata={
                            "artifact_index": index,
                            "component": component,
                        },
                    )
                )

        return issues

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True

        if isinstance(value, str):
            return not value.strip()

        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0

        return False

    @staticmethod
    def _issue(
        *,
        code: str,
        message: str,
        severity: EvaluationSeverity = EvaluationSeverity.ERROR,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationIssue:
        return EvaluationIssue(
            code=code,
            message=message,
            severity=severity,
            check_type=EvaluationCheckType.COMPLETENESS,
            metadata=metadata or {},
        )