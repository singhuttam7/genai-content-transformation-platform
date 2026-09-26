from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.transformation.contracts import (
    TransformationType,
)
from app.schemas.transformation import (
    TransformationCreateRequest,
    TransformationListResponse,
    TransformationResponse,
)


def test_create_request_accepts_valid_data() -> None:
    project_id = uuid4()
    source_id = uuid4()

    request = TransformationCreateRequest(
        project_id=project_id,
        source_id=source_id,
        transformation_type=TransformationType.ADVISORY,
        objective="Create a professional advisory.",
        audience="General public",
        tone="Professional",
        language="English",
        requested_outputs=["advisory"],
        configuration={"priority": "normal"},
    )

    assert request.project_id == project_id
    assert request.source_id == source_id
    assert (
        request.transformation_type
        == TransformationType.ADVISORY
    )


def test_create_request_defaults_are_deterministic() -> None:
    request = TransformationCreateRequest(
        project_id=uuid4(),
        source_id=uuid4(),
        transformation_type=TransformationType.VIDEO,
    )

    assert request.language == "English"
    assert request.requested_outputs == []
    assert request.configuration == {}


def test_create_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TransformationCreateRequest(
            project_id=uuid4(),
            source_id=uuid4(),
            transformation_type=TransformationType.ADVISORY,
            unknown_field="not allowed",
        )


def test_create_request_rejects_invalid_transformation_type() -> None:
    with pytest.raises(ValidationError):
        TransformationCreateRequest(
            project_id=uuid4(),
            source_id=uuid4(),
            transformation_type="unsupported",
        )


def test_create_request_requires_project_and_source() -> None:
    with pytest.raises(ValidationError):
        TransformationCreateRequest(
            transformation_type=TransformationType.ADVISORY,
        )


def test_response_supports_orm_objects() -> None:
    now = datetime.now(timezone.utc)
    project_id = uuid4()
    source_id = uuid4()
    transformation_id = uuid4()

    class FakeTransformation:
        pass

    fake = FakeTransformation()
    fake.id = transformation_id
    fake.project_id = project_id
    fake.source_id = source_id
    fake.objective = "Create advisory"
    fake.audience = "Developers"
    fake.tone = "Professional"
    fake.language = "English"
    fake.detail_level = "Detailed"
    fake.style = "Formal"
    fake.requested_outputs = ["advisory"]
    fake.configuration = {"priority": "normal"}
    fake.status = "DRAFT"
    fake.created_at = now
    fake.updated_at = now

    response = TransformationResponse.model_validate(fake)

    assert response.id == transformation_id
    assert response.project_id == project_id
    assert response.source_id == source_id
    assert response.status == "DRAFT"


def test_list_response_accepts_items() -> None:
    now = datetime.now(timezone.utc)
    transformation_id = uuid4()
    project_id = uuid4()
    source_id = uuid4()

    response = TransformationResponse(
        id=transformation_id,
        project_id=project_id,
        source_id=source_id,
        objective=None,
        audience=None,
        tone=None,
        language="English",
        detail_level=None,
        style=None,
        requested_outputs=[],
        configuration={},
        status="DRAFT",
        created_at=now,
        updated_at=now,
    )

    collection = TransformationListResponse(
        items=[response],
        total=1,
    )

    assert collection.total == 1
    assert len(collection.items) == 1