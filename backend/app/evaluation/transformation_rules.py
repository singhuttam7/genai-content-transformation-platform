from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.agents.transformation.contracts import (
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


class TransformationRulesEvaluator(EvaluationPort):
    """Validate transformation-type-specific output rules."""

    @property
    def check_type(self) -> EvaluationCheckType:
        return EvaluationCheckType.TRANSFORMATION

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest.")

        transformation_type = (
            request.transformation_request.transformation_type
        )

        issues: list[EvaluationIssue] = []

        for index, artifact in enumerate(
            request.transformation_result.artifacts
        ):
            issues.extend(
                self._validate_artifact(
                    transformation_type=transformation_type,
                    content=artifact.content,
                    index=index,
                )
            )

        status = (
            EvaluationStatus.FAILED
            if issues
            else EvaluationStatus.PASSED
        )

        score = 0.0 if issues else 1.0

        check = EvaluationCheck(
            check_type=EvaluationCheckType.TRANSFORMATION,
            status=status,
            score=score,
            issues=issues,
            metadata={
                "transformation_type": transformation_type.value,
                "artifact_count": len(
                    request.transformation_result.artifacts
                ),
            },
        )

        return EvaluationResult(
            status=status,
            score=score,
            checks=[check],
            issues=issues,
            metadata={
                "evaluator": self.__class__.__name__,
                "transformation_type": transformation_type.value,
                "artifact_count": len(
                    request.transformation_result.artifacts
                ),
            },
        )

    @classmethod
    def _validate_artifact(
        cls,
        *,
        transformation_type: TransformationType,
        content: Any,
        index: int,
    ) -> list[EvaluationIssue]:
        if not isinstance(content, Mapping):
            return [
                cls._issue(
                    code=f"ARTIFACT_{index}_STRUCTURED_CONTENT_REQUIRED",
                    message=(
                        "Transformation-specific validation requires "
                        "structured artifact content."
                    ),
                    metadata={"artifact_index": index},
                )
            ]

        validators = {
            TransformationType.ADVISORY: cls._validate_advisory,
            TransformationType.EXECUTIVE_SUMMARY: (
                cls._validate_executive_summary
            ),
            TransformationType.SOCIAL_MEDIA: cls._validate_social_media,
            TransformationType.INFOGRAPHIC: cls._validate_infographic,
            TransformationType.PRESENTATION: cls._validate_presentation,
            TransformationType.VIDEO: cls._validate_video,
        }

        return validators[transformation_type](
            content,
            index,
        )

    @classmethod
    def _validate_advisory(
        cls,
        content: Mapping[str, Any],
        index: int,
    ) -> list[EvaluationIssue]:
        issues: list[EvaluationIssue] = []

        cls._require_non_empty_string(
            content,
            "title",
            index,
            issues,
        )

        recommendations = content.get("recommendations")

        if isinstance(recommendations, list):
            if not recommendations:
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_ADVISORY_NO_RECOMMENDATIONS",
                        message=(
                            "An advisory must contain at least one "
                            "recommendation."
                        ),
                        metadata={"artifact_index": index},
                    )
                )

        return issues

    @classmethod
    def _validate_executive_summary(
        cls,
        content: Mapping[str, Any],
        index: int,
    ) -> list[EvaluationIssue]:
        issues: list[EvaluationIssue] = []

        cls._require_non_empty_string(
            content,
            "summary",
            index,
            issues,
        )

        findings = content.get("key_findings")

        if isinstance(findings, list) and not findings:
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_SUMMARY_NO_FINDINGS",
                    message=(
                        "An executive summary must contain at least "
                        "one key finding."
                    ),
                    metadata={"artifact_index": index},
                )
            )

        return issues

    @classmethod
    def _validate_social_media(
        cls,
        content: Mapping[str, Any],
        index: int,
    ) -> list[EvaluationIssue]:
        issues: list[EvaluationIssue] = []

        platform = content.get("platform")

        if not cls._is_non_empty(platform):
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_SOCIAL_PLATFORM_MISSING",
                    message="A social media artifact must specify a platform.",
                    metadata={"artifact_index": index},
                )
            )

        post_content = content.get("content")

        if isinstance(post_content, str):
            if len(post_content.strip()) < 10:
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_SOCIAL_CONTENT_TOO_SHORT",
                        message=(
                            "Social media content must contain meaningful "
                            "publication text."
                        ),
                        metadata={"artifact_index": index},
                    )
                )

        return issues

    @classmethod
    def _validate_infographic(
        cls,
        content: Mapping[str, Any],
        index: int,
    ) -> list[EvaluationIssue]:
        issues: list[EvaluationIssue] = []

        sections = content.get("sections")

        if isinstance(sections, list):
            if len(sections) < 1:
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_INFOGRAPHIC_NO_SECTIONS",
                        message=(
                            "An infographic must contain at least "
                            "one section."
                        ),
                        metadata={"artifact_index": index},
                    )
                )

        visual_recommendations = content.get(
            "visual_recommendations"
        )

        if isinstance(visual_recommendations, list):
            if not visual_recommendations:
                issues.append(
                    cls._issue(
                        code=(
                            f"ARTIFACT_{index}_"
                            "INFOGRAPHIC_NO_VISUAL_RECOMMENDATIONS"
                        ),
                        message=(
                            "An infographic must contain visual "
                            "recommendations."
                        ),
                        metadata={"artifact_index": index},
                    )
                )

        return issues

    @classmethod
    def _validate_presentation(
        cls,
        content: Mapping[str, Any],
        index: int,
    ) -> list[EvaluationIssue]:
        issues: list[EvaluationIssue] = []

        slides = content.get("slides")

        if isinstance(slides, list) and len(slides) < 2:
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_PRESENTATION_TOO_FEW_SLIDES",
                    message=(
                        "A presentation must contain at least "
                        "two slides."
                    ),
                    metadata={"artifact_index": index},
                )
            )

        speaker_notes = content.get("speaker_notes")

        if isinstance(speaker_notes, list):
            if not speaker_notes:
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_PRESENTATION_NO_NOTES",
                        message=(
                            "A presentation must contain speaker notes."
                        ),
                        metadata={"artifact_index": index},
                    )
                )

        return issues

    @classmethod
    def _validate_video(
        cls,
        content: Mapping[str, Any],
        index: int,
    ) -> list[EvaluationIssue]:
        issues: list[EvaluationIssue] = []

        scenes = content.get("scenes")

        if isinstance(scenes, list) and not scenes:
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_VIDEO_NO_SCENES",
                    message="A video package must contain scenes.",
                    metadata={"artifact_index": index},
                )
            )

        narration = content.get("narration")

        if isinstance(narration, str) and not narration.strip():
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_VIDEO_NO_NARRATION",
                    message="A video package must contain narration.",
                    metadata={"artifact_index": index},
                )
            )

        subtitles = content.get("subtitles")

        if isinstance(subtitles, list) and not subtitles:
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_VIDEO_NO_SUBTITLES",
                    message="A video package must contain subtitles.",
                    metadata={"artifact_index": index},
                )
            )

        return issues

    @staticmethod
    def _require_non_empty_string(
        content: Mapping[str, Any],
        field: str,
        index: int,
        issues: list[EvaluationIssue],
    ) -> None:
        value = content.get(field)

        if not isinstance(value, str) or not value.strip():
            issues.append(
                TransformationRulesEvaluator._issue(
                    code=f"ARTIFACT_{index}_MISSING_{field.upper()}",
                    message=(
                        f"Transformation output must contain a "
                        f"non-empty '{field}'."
                    ),
                    metadata={
                        "artifact_index": index,
                        "field": field,
                    },
                )
            )

    @staticmethod
    def _is_non_empty(value: Any) -> bool:
        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)

        return True

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
            check_type=EvaluationCheckType.TRANSFORMATION,
            metadata=metadata or {},
        )