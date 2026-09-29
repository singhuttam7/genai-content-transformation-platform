from uuid import uuid4

import pytest

from app.agents.transformation.contracts import ArtifactEnvelope
from app.artifacts.service import ArtifactPersistenceService


def test_artifact_persistence_service_requires_async_session():
    with pytest.raises(TypeError, match="session must be an AsyncSession"):
        ArtifactPersistenceService(session=object())


def test_artifact_persistence_service_imports():
    assert ArtifactPersistenceService is not None


def test_artifact_envelope_can_be_created():
    envelope = ArtifactEnvelope(
        artifact_type="executive_summary",
        title="Test Summary",
        content="Generated summary content.",
        metadata={},
    )

    assert envelope.artifact_type == "executive_summary"
    assert envelope.title == "Test Summary"
    assert envelope.content == "Generated summary content."