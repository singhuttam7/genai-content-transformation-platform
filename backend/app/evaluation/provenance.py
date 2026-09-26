from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.agents.transformation.contracts import ArtifactEnvelope
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


class ProvenanceEvaluator(EvaluationPort):
    """Validate provenance information attached to transformation artifacts."""

    @property
    def check_type(self) -> EvaluationCheckType:
        return EvaluationCheckType.PROVENANCE

    async def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest.")

        issues: list[EvaluationIssue] = []

        if not request.transformation_result.artifacts:
            issues.append(
                self._issue(
                    code="NO_ARTIFACTS",
                    message="The transformation result contains no artifacts.",
                )
            )

        for index, artifact in enumerate(
            request.transformation_result.artifacts
        ):
            issues.extend(
                self._validate_artifact(
                    artifact=artifact,
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
            check_type=EvaluationCheckType.PROVENANCE,
            status=status,
            score=score,
            issues=issues,
            metadata={
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
                "artifact_count": len(
                    request.transformation_result.artifacts
                ),
            },
        )

    @classmethod
    def _validate_artifact(
        cls,
        *,
        artifact: ArtifactEnvelope,
        index: int,
    ) -> list[EvaluationIssue]:
        provenance = artifact.provenance

        if not isinstance(provenance, Mapping):
            return [
                cls._issue(
                    code=f"ARTIFACT_{index}_PROVENANCE_INVALID",
                    message="Artifact provenance must be a mapping.",
                    metadata={"artifact_index": index},
                )
            ]

        if not provenance:
            return [
                cls._issue(
                    code=f"ARTIFACT_{index}_PROVENANCE_MISSING",
                    message=(
                        "Artifact provenance is missing. "
                        "Generated content must retain provenance."
                    ),
                    metadata={"artifact_index": index},
                )
            ]

        issues: list[EvaluationIssue] = []

        source_keys = (
            "source",
            "source_id",
            "source_ids",
            "source_uri",
            "sources",
            "document_id",
            "document_ids",
            "chunk_id",
            "chunk_ids",
            "reference",
            "references",
        )

        if not any(
            key in provenance and not cls._is_empty(provenance[key])
            for key in source_keys
        ):
            issues.append(
                cls._issue(
                    code=f"ARTIFACT_{index}_SOURCE_REFERENCE_MISSING",
                    message=(
                        "Artifact provenance does not contain a usable "
                        "source or reference."
                    ),
                    metadata={
                        "artifact_index": index,
                    },
                )
            )

        if "sources" in provenance:
            sources = provenance["sources"]

            if sources is not None and not isinstance(
                sources,
                (str, list, tuple, set, Mapping),
            ):
                issues.append(
                    cls._issue(
                        code=f"ARTIFACT_{index}_SOURCES_INVALID",
                        message="The provenance sources field has an invalid type.",
                        metadata={"artifact_index": index},
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
            check_type=EvaluationCheckType.PROVENANCE,
            metadata=metadata or {},
        )