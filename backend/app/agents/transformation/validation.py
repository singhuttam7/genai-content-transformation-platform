from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
)


class ValidationStatus(StrEnum):
    VALID = "valid"
    REVISION_REQUIRED = "revision_required"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: str = Field(default="error", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus
    issues: list[ValidationIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if (
            self.status == ValidationStatus.VALID
            and self.issues
        ):
            raise ValueError(
                "Valid results must not contain validation issues."
            )

        if (
            self.status == ValidationStatus.REVISION_REQUIRED
            and not self.issues
        ):
            raise ValueError(
                "Revision-required results must contain "
                "at least one validation issue."
            )


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_request: TransformationRequest
    previous_result: TransformationResult
    issues: list[ValidationIssue] = Field(
        min_length=1,
    )
    instructions: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransformationValidationHook(ABC):
    """
    Provider-independent hook for validating transformation output.

    Concrete validation policies belong to the validation/evaluation
    subsystem. A8 only defines the integration contract.
    """

    @abstractmethod
    async def validate(
        self,
        *,
        request: TransformationRequest,
        result: TransformationResult,
    ) -> ValidationResult:
        raise NotImplementedError


class TransformationRevisionHook(ABC):
    """
    Provider-independent hook for requesting a transformation revision.

    The hook does not itself implement LLM generation. It defines the
    contract through which orchestration can hand validation feedback
    back to a transformation agent.
    """

    @abstractmethod
    async def revise(
        self,
        request: RevisionRequest,
    ) -> TransformationResult:
        raise NotImplementedError


class ValidationRevisionCoordinator:
    """
    Coordinates validation and optional revision without owning the
    validation policy or transformation implementation.

    A single coordinator invocation performs at most one validation
    and, when required, one revision attempt.
    """

    def __init__(
        self,
        *,
        validator: TransformationValidationHook,
        reviser: TransformationRevisionHook,
    ) -> None:
        if not isinstance(
            validator,
            TransformationValidationHook,
        ):
            raise TypeError(
                "validator must be a "
                "TransformationValidationHook."
            )

        if not isinstance(
            reviser,
            TransformationRevisionHook,
        ):
            raise TypeError(
                "reviser must be a "
                "TransformationRevisionHook."
            )

        self._validator = validator
        self._reviser = reviser

    async def process(
        self,
        *,
        request: TransformationRequest,
        result: TransformationResult,
    ) -> TransformationResult:
        validation = await self._validator.validate(
            request=request,
            result=result,
        )

        if validation.status == ValidationStatus.VALID:
            return result

        revision_request = RevisionRequest(
            original_request=request,
            previous_result=result,
            issues=validation.issues,
            instructions=self._build_revision_instructions(
                validation
            ),
            metadata={
                "validation_status": validation.status.value,
                **validation.metadata,
            },
        )

        return await self._reviser.revise(
            revision_request
        )

    @staticmethod
    def _build_revision_instructions(
        validation: ValidationResult,
    ) -> str:
        lines = [
            "Revise the transformation output using "
            "the following validation feedback:"
        ]

        for index, issue in enumerate(
            validation.issues,
            start=1,
        ):
            lines.append(
                f"{index}. [{issue.code}] "
                f"{issue.message}"
            )

        return "\n".join(lines)


class PassthroughValidationHook(
    TransformationValidationHook,
):
    """
    Minimal default hook used when no validation policy is configured.

    It intentionally performs no quality judgment. Actual validation
    belongs to A9.
    """

    async def validate(
        self,
        *,
        request: TransformationRequest,
        result: TransformationResult,
    ) -> ValidationResult:
        return ValidationResult(
            status=ValidationStatus.VALID,
        )


class TransformationRevisionAdapter(
    TransformationRevisionHook,
):
    """
    Adapter around a transformation agent.

    The adapter converts revision feedback into a normal
    TransformationRequest while preserving the original request.
    """

    def __init__(self, agent: Any) -> None:
        if not hasattr(agent, "execute"):
            raise TypeError(
                "agent must provide an execute method."
            )

        self._agent = agent

    async def revise(
        self,
        request: RevisionRequest,
    ) -> TransformationResult:
        configuration = dict(
            request.original_request.configuration
        )

        configuration["revision"] = {
            "instructions": request.instructions,
            "issues": [
                issue.model_dump()
                for issue in request.issues
            ],
        }

        revised_request = request.original_request.model_copy(
            update={
                "configuration": configuration,
                "metadata": {
                    **request.original_request.metadata,
                    "revision_requested": True,
                    "revision_issue_count": len(
                        request.issues
                    ),
                },
            }
        )

        return await self._agent.execute(
            revised_request
        )