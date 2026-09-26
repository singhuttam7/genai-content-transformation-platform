from __future__ import annotations

from typing import Any

from app.agents.transformation.contracts import (
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
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


class StructuralEvaluator(EvaluationPort):
    """Validate the basic structural integrity of transformation output."""

    @property
    def check_type(self) -> EvaluationCheckType:
        return EvaluationCheckType.STRUCTURE

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        if not isinstance(request, EvaluationRequest):
            raise TypeError(
                "request must be an EvaluationRequest."
            )

        issues: list[EvaluationIssue] = []

        transformation_request = request.transformation_request
        transformation_result = request.transformation_result

        if not isinstance(
            transformation_request,
            TransformationRequest,
        ):
            raise TypeError(
                "transformation_request must be a "
                "TransformationRequest."
            )

        if not isinstance(
            transformation_result,
            TransformationResult,
        ):
            raise TypeError(
                "transformation_result must be a "
                "TransformationResult."
            )

        if transformation_result.status == TransformationStatus.FAILED:
            issues.append(
                self._issue(
                    code="TRANSFORMATION_FAILED",
                    message=(
                        "The transformation result is marked as failed."
                    ),
                    severity=EvaluationSeverity.ERROR,
                )
            )

        if not transformation_result.artifacts:
            issues.append(
                self._issue(
                    code="NO_ARTIFACTS",
                    message=(
                        "The transformation result contains no artifacts."
                    ),
                )
            )

        for index, artifact in enumerate(
            transformation_result.artifacts
        ):
            prefix = f"ARTIFACT_{index}"

            if not artifact.artifact_type.strip():
                issues.append(
                    self._issue(
                        code=f"{prefix}_TYPE_MISSING",
                        message=(
                            "Artifact type must be a non-empty string."
                        ),
                    )
                )

            if artifact.content is None:
                issues.append(
                    self._issue(
                        code=f"{prefix}_CONTENT_MISSING",
                        message=(
                            "Artifact content must not be None."
                        ),
                    )
                )
            elif isinstance(artifact.content, str):
                if not artifact.content.strip():
                    issues.append(
                        self._issue(
                            code=f"{prefix}_CONTENT_EMPTY",
                            message=(
                                "Artifact string content must not "
                                "be empty."
                            ),
                        )
                    )

            if not isinstance(artifact.metadata, dict):
                issues.append(
                    self._issue(
                        code=f"{prefix}_METADATA_INVALID",
                        message="Artifact metadata must be a dictionary.",
                    )
                )

            if not isinstance(artifact.provenance, dict):
                issues.append(
                    self._issue(
                        code=f"{prefix}_PROVENANCE_INVALID",
                        message=(
                            "Artifact provenance must be a dictionary."
                        ),
                    )
                )

        if request.artifacts:
            if len(request.artifacts) != len(
                transformation_result.artifacts
            ):
                issues.append(
                    self._issue(
                        code="ARTIFACT_REFERENCE_MISMATCH",
                        message=(
                            "Evaluation artifact references do not "
                            "match the transformation result."
                        ),
                    )
                )

        status = (
            EvaluationStatus.FAILED
            if issues
            else EvaluationStatus.PASSED
        )

        check = EvaluationCheck(
            check_type=EvaluationCheckType.STRUCTURE,
            status=status,
            score=(
                1.0
                if not issues
                else 0.0
            ),
            issues=issues,
            metadata={
                "artifact_count": len(
                    transformation_result.artifacts
                ),
            },
        )

        return EvaluationResult(
            status=status,
            score=check.score,
            checks=[check],
            issues=issues,
            metadata={
                "evaluator": self.__class__.__name__,
                "artifact_count": len(
                    transformation_result.artifacts
                ),
            },
        )

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
            check_type=EvaluationCheckType.STRUCTURE,
            metadata=metadata or {},
        )