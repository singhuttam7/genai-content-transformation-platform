from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)
from app.agents.transformation.validation import (
    PassthroughValidationHook,
    RevisionRequest,
    TransformationRevisionAdapter,
    TransformationRevisionHook,
    TransformationValidationHook,
    ValidationIssue,
    ValidationResult,
    ValidationRevisionCoordinator,
    ValidationStatus,
)


def create_request() -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source material",
        metadata={"source": "test"},
    )


def create_result() -> TransformationResult:
    return TransformationResult(
        status=TransformationStatus.COMPLETED,
        artifacts=[
            ArtifactEnvelope(
                artifact_type="advisory",
                content="Generated content",
            )
        ],
    )


def create_issue() -> ValidationIssue:
    return ValidationIssue(
        code="MISSING_DETAIL",
        message="Important detail is missing.",
    )


def test_validation_status_values() -> None:
    assert ValidationStatus.VALID.value == "valid"
    assert (
        ValidationStatus.REVISION_REQUIRED.value
        == "revision_required"
    )


def test_validation_issue_defaults() -> None:
    issue = create_issue()

    assert issue.severity == "error"
    assert issue.metadata == {}


def test_validation_issue_rejects_blank_code() -> None:
    with pytest.raises(ValueError):
        ValidationIssue(
            code="",
            message="Issue",
        )


def test_validation_result_valid_has_no_issues() -> None:
    result = ValidationResult(
        status=ValidationStatus.VALID,
    )

    assert result.issues == []


def test_validation_result_valid_rejects_issues() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain validation issues",
    ):
        ValidationResult(
            status=ValidationStatus.VALID,
            issues=[create_issue()],
        )


def test_validation_result_revision_requires_issue() -> None:
    with pytest.raises(
        ValueError,
        match="must contain",
    ):
        ValidationResult(
            status=ValidationStatus.REVISION_REQUIRED,
        )


def test_revision_request_requires_issue() -> None:
    request = RevisionRequest(
        original_request=create_request(),
        previous_result=create_result(),
        issues=[create_issue()],
        instructions="Fix the missing detail.",
    )

    assert len(request.issues) == 1


def test_revision_request_rejects_empty_issues() -> None:
    with pytest.raises(ValueError):
        RevisionRequest(
            original_request=create_request(),
            previous_result=create_result(),
            issues=[],
            instructions="Fix the output.",
        )


def test_validation_hook_is_abstract() -> None:
    with pytest.raises(TypeError):
        TransformationValidationHook()


def test_revision_hook_is_abstract() -> None:
    with pytest.raises(TypeError):
        TransformationRevisionHook()


@pytest.mark.asyncio
async def test_passthrough_validator_returns_valid() -> None:
    validator = PassthroughValidationHook()

    result = await validator.validate(
        request=create_request(),
        result=create_result(),
    )

    assert result.status == ValidationStatus.VALID
    assert result.issues == []


def test_coordinator_rejects_invalid_validator() -> None:
    class FakeRevisioner(TransformationRevisionHook):
        async def revise(
            self,
            request: RevisionRequest,
        ) -> TransformationResult:
            return request.previous_result

    with pytest.raises(TypeError):
        ValidationRevisionCoordinator(
            validator=object(),  # type: ignore[arg-type]
            reviser=FakeRevisioner(),
        )


def test_coordinator_rejects_invalid_reviser() -> None:
    class FakeValidator(TransformationValidationHook):
        async def validate(
            self,
            *,
            request: TransformationRequest,
            result: TransformationResult,
        ) -> ValidationResult:
            return ValidationResult(
                status=ValidationStatus.VALID
            )

    with pytest.raises(TypeError):
        ValidationRevisionCoordinator(
            validator=FakeValidator(),
            reviser=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_valid_output_skips_revision() -> None:
    class Validator(TransformationValidationHook):
        async def validate(
            self,
            *,
            request: TransformationRequest,
            result: TransformationResult,
        ) -> ValidationResult:
            return ValidationResult(
                status=ValidationStatus.VALID
            )

    reviser = AsyncMock(
        spec=TransformationRevisionHook
    )

    class Reviser(TransformationRevisionHook):
        async def revise(
            self,
            request: RevisionRequest,
        ) -> TransformationResult:
            return await reviser.revise(request)

    coordinator = ValidationRevisionCoordinator(
        validator=Validator(),
        reviser=Reviser(),
    )

    original = create_result()

    returned = await coordinator.process(
        request=create_request(),
        result=original,
    )

    assert returned is original
    reviser.revise.assert_not_awaited()


@pytest.mark.asyncio
async def test_revision_is_requested_when_validation_fails() -> None:
    issue = create_issue()

    class Validator(TransformationValidationHook):
        async def validate(
            self,
            *,
            request: TransformationRequest,
            result: TransformationResult,
        ) -> ValidationResult:
            return ValidationResult(
                status=ValidationStatus.REVISION_REQUIRED,
                issues=[issue],
                metadata={"validator": "test"},
            )

    captured: list[RevisionRequest] = []

    class Reviser(TransformationRevisionHook):
        async def revise(
            self,
            request: RevisionRequest,
        ) -> TransformationResult:
            captured.append(request)
            return create_result()

    coordinator = ValidationRevisionCoordinator(
        validator=Validator(),
        reviser=Reviser(),
    )

    returned = await coordinator.process(
        request=create_request(),
        result=create_result(),
    )

    assert returned.status == TransformationStatus.COMPLETED
    assert len(captured) == 1
    assert captured[0].issues == [issue]


@pytest.mark.asyncio
async def test_revision_instructions_contain_feedback() -> None:
    class Validator(TransformationValidationHook):
        async def validate(
            self,
            *,
            request: TransformationRequest,
            result: TransformationResult,
        ) -> ValidationResult:
            return ValidationResult(
                status=ValidationStatus.REVISION_REQUIRED,
                issues=[
                    ValidationIssue(
                        code="FACTUAL",
                        message="Check factual consistency.",
                    ),
                    ValidationIssue(
                        code="STRUCTURE",
                        message="Improve structure.",
                    ),
                ],
            )

    captured: list[RevisionRequest] = []

    class Reviser(TransformationRevisionHook):
        async def revise(
            self,
            request: RevisionRequest,
        ) -> TransformationResult:
            captured.append(request)
            return create_result()

    await ValidationRevisionCoordinator(
        validator=Validator(),
        reviser=Reviser(),
    ).process(
        request=create_request(),
        result=create_result(),
    )

    instructions = captured[0].instructions

    assert "FACTUAL" in instructions
    assert "Check factual consistency." in instructions
    assert "STRUCTURE" in instructions
    assert "Improve structure." in instructions


@pytest.mark.asyncio
async def test_validation_metadata_is_forwarded() -> None:
    class Validator(TransformationValidationHook):
        async def validate(
            self,
            *,
            request: TransformationRequest,
            result: TransformationResult,
        ) -> ValidationResult:
            return ValidationResult(
                status=ValidationStatus.REVISION_REQUIRED,
                issues=[create_issue()],
                metadata={"score": 0.5},
            )

    captured: list[RevisionRequest] = []

    class Reviser(TransformationRevisionHook):
        async def revise(
            self,
            request: RevisionRequest,
        ) -> TransformationResult:
            captured.append(request)
            return create_result()

    await ValidationRevisionCoordinator(
        validator=Validator(),
        reviser=Reviser(),
    ).process(
        request=create_request(),
        result=create_result(),
    )

    assert captured[0].metadata["score"] == 0.5
    assert (
        captured[0].metadata["validation_status"]
        == "revision_required"
    )


def test_revision_adapter_requires_execute() -> None:
    with pytest.raises(TypeError):
        TransformationRevisionAdapter(object())


@pytest.mark.asyncio
async def test_revision_adapter_calls_agent() -> None:
    agent = AsyncMock()

    expected = create_result()

    agent.execute.return_value = expected

    adapter = TransformationRevisionAdapter(agent)

    revision = RevisionRequest(
        original_request=create_request(),
        previous_result=create_result(),
        issues=[create_issue()],
        instructions="Fix the issue.",
    )

    result = await adapter.revise(revision)

    assert result is expected
    agent.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_adapter_adds_revision_configuration() -> None:
    agent = AsyncMock()
    agent.execute.return_value = create_result()

    adapter = TransformationRevisionAdapter(agent)

    revision = RevisionRequest(
        original_request=create_request(),
        previous_result=create_result(),
        issues=[create_issue()],
        instructions="Fix the issue.",
    )

    await adapter.revise(revision)

    revised_request = agent.execute.call_args.args[0]

    assert (
        revised_request.configuration["revision"][
            "instructions"
        ]
        == "Fix the issue."
    )


@pytest.mark.asyncio
async def test_revision_adapter_preserves_original_configuration() -> None:
    agent = AsyncMock()
    agent.execute.return_value = create_result()

    original = TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Source",
        configuration={
            "format": "formal",
        },
    )

    revision = RevisionRequest(
        original_request=original,
        previous_result=create_result(),
        issues=[create_issue()],
        instructions="Fix the issue.",
    )

    await TransformationRevisionAdapter(
        agent
    ).revise(revision)

    revised_request = agent.execute.call_args.args[0]

    assert (
        revised_request.configuration["format"]
        == "formal"
    )


@pytest.mark.asyncio
async def test_revision_adapter_marks_metadata() -> None:
    agent = AsyncMock()
    agent.execute.return_value = create_result()

    revision = RevisionRequest(
        original_request=create_request(),
        previous_result=create_result(),
        issues=[
            create_issue(),
            ValidationIssue(
                code="STYLE",
                message="Adjust style.",
            ),
        ],
        instructions="Fix issues.",
    )

    await TransformationRevisionAdapter(
        agent
    ).revise(revision)

    revised_request = agent.execute.call_args.args[0]

    assert revised_request.metadata["revision_requested"] is True
    assert revised_request.metadata["revision_issue_count"] == 2