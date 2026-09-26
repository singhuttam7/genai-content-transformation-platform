from __future__ import annotations

import pytest

from app.agents.transformation.contracts import (
    ArtifactEnvelope,
    TransformationRequest,
    TransformationResult,
    TransformationType,
)
from app.evaluation.contracts import (
    EvaluationCheckType,
    EvaluationRequest,
    EvaluationStatus,
)
from app.evaluation.provenance import ProvenanceEvaluator


def create_request() -> TransformationRequest:
    return TransformationRequest(
        transformation_type=TransformationType.ADVISORY,
        input="Test source",
    )


def create_evaluation_request(
    provenance: dict,
    *,
    content: str = "Generated content",
) -> EvaluationRequest:
    return EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content=content,
                    provenance=provenance,
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_exposes_provenance_check_type() -> None:
    assert (
        ProvenanceEvaluator().check_type
        == EvaluationCheckType.PROVENANCE
    )


@pytest.mark.asyncio
async def test_source_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"source": "document-001"}
        )
    )

    assert result.status == EvaluationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []


@pytest.mark.asyncio
async def test_source_id_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"source_id": "doc-001"}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_source_uri_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"source_uri": "https://example.com/source"}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_sources_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"sources": ["doc-001", "doc-002"]}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_document_id_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"document_id": "document-001"}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_chunk_id_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"chunk_id": "chunk-001"}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_reference_provenance_passes() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"reference": "ref-001"}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_empty_provenance_fails() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request({})
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_PROVENANCE_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_empty_source_reference_fails() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"source": ""}
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SOURCE_REFERENCE_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_empty_source_list_fails() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"sources": []}
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SOURCE_REFERENCE_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_metadata_without_source_reference_fails() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {
                "agent": "advisory_agent",
                "model": "test-model",
            }
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SOURCE_REFERENCE_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_multiple_provenance_fields_pass() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {
                "source_id": "doc-001",
                "document_id": "document-001",
                "chunk_ids": ["chunk-001", "chunk-002"],
                "reference": "section-1",
            }
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_sources_string_is_allowed() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"sources": "document-001"}
        )
    )

    assert result.status == EvaluationStatus.PASSED


@pytest.mark.asyncio
async def test_evaluator_metadata_is_present() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"source": "document-001"}
        )
    )

    assert (
        result.metadata["evaluator"]
        == "ProvenanceEvaluator"
    )
    assert result.metadata["artifact_count"] == 1


@pytest.mark.asyncio
async def test_check_metadata_contains_artifact_count() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"source": "document-001"}
        )
    )

    assert len(result.checks) == 1
    assert (
        result.checks[0].check_type
        == EvaluationCheckType.PROVENANCE
    )
    assert result.checks[0].metadata["artifact_count"] == 1


@pytest.mark.asyncio
async def test_no_artifacts_fails() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[],
        ),
    )

    result = await ProvenanceEvaluator().evaluate(request)

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "NO_ARTIFACTS"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_multiple_artifacts_are_checked() -> None:
    request = EvaluationRequest(
        transformation_request=create_request(),
        transformation_result=TransformationResult(
            status="completed",
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    content="First",
                    provenance={"source_id": "doc-001"},
                ),
                ArtifactEnvelope(
                    artifact_type="summary",
                    content="Second",
                    provenance={},
                ),
            ],
        ),
    )

    result = await ProvenanceEvaluator().evaluate(request)

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_1_PROVENANCE_MISSING"
        for issue in result.issues
    )


@pytest.mark.asyncio
async def test_invalid_sources_type_is_reported() -> None:
    result = await ProvenanceEvaluator().evaluate(
        create_evaluation_request(
            {"sources": 123}
        )
    )

    assert result.status == EvaluationStatus.FAILED
    assert any(
        issue.code == "ARTIFACT_0_SOURCES_INVALID"
        for issue in result.issues
    )