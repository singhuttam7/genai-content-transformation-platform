from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
)


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class EvaluationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EvaluationCheckType(StrEnum):
    STRUCTURE = "structure"
    COMPLETENESS = "completeness"
    PROVENANCE = "provenance"
    FACTUAL = "factual"
    TRANSFORMATION = "transformation"
    QUALITY = "quality"


class EvaluationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: EvaluationSeverity = EvaluationSeverity.ERROR
    check_type: EvaluationCheckType
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_type: EvaluationCheckType
    status: EvaluationStatus
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    issues: list[EvaluationIssue] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transformation_request: TransformationRequest
    transformation_result: TransformationResult
    artifacts: list[ArtifactEnvelope] = Field(
        default_factory=list
    )
    checks: list[EvaluationCheckType] = Field(
        default_factory=lambda: list(
            EvaluationCheckType
        )
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: EvaluationStatus
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    checks: list[EvaluationCheck] = Field(
        default_factory=list
    )
    issues: list[EvaluationIssue] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if (
            self.status == EvaluationStatus.PASSED
            and any(
                issue.severity
                == EvaluationSeverity.ERROR
                for issue in self.issues
            )
        ):
            raise ValueError(
                "Passed evaluation results cannot contain "
                "error-severity issues."
            )

        if self.score is not None:
            if not 0.0 <= self.score <= 1.0:
                raise ValueError(
                    "Evaluation score must be between 0 and 1."
                )