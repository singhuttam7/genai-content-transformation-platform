from __future__ import annotations

from uuid import UUID

from app.knowledge.chunking.schemas import KnowledgeChunkDraft
from app.knowledge.normalization.schemas import (
    NormalizedKnowledgeDocument,
)
from app.knowledge.provenance.metadata_builder import (
    SourceMetadataBuilder,
)
from app.knowledge.provenance.schemas import (
    KnowledgeChunkMetadata,
    KnowledgeChunkProvenance,
    KnowledgeDocumentProvenance,
    KnowledgeLocationProvenance,
    KnowledgeSourceProvenance,
)


class KnowledgeProvenanceBuilder:
    """
    Deterministically construct complete provenance metadata for
    a knowledge chunk.

    This component:
    - does not access the database
    - does not generate IDs
    - does not call an LLM
    - does not call embedding models
    - does not access a vector store
    - does not mutate the input objects

    Identity information such as project_id and document_id must be
    supplied by the caller.

    Source-specific metadata is delegated to SourceMetadataBuilder.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        source_metadata_builder: SourceMetadataBuilder | None = None,
    ) -> None:
        self.source_metadata_builder = (
            source_metadata_builder
            or SourceMetadataBuilder()
        )

    def build(
        self,
        *,
        document: NormalizedKnowledgeDocument,
        chunk: KnowledgeChunkDraft,
        project_id: UUID,
        document_id: UUID,
        document_version: int,
    ) -> KnowledgeChunkMetadata:
        """
        Build complete metadata and provenance for one chunk.
        """

        source = self._build_source_provenance(
            document=document,
            project_id=project_id,
        )

        document_provenance = (
            self._build_document_provenance(
                document=document,
                document_id=document_id,
                document_version=document_version,
            )
        )

        location = self._build_location_provenance(
            chunk=chunk,
        )

        provenance = KnowledgeChunkProvenance(
            source=source,
            document=document_provenance,
            location=location,
            chunk_index=chunk.chunk_index,
        )

        enriched_metadata = (
            self.source_metadata_builder.build(
                document=document,
            )
        )

        metadata = dict(chunk.metadata)

        metadata["source_metadata"] = (
            enriched_metadata.source_metadata.model_dump(
                mode="json"
            )
        )

        return KnowledgeChunkMetadata(
            chunk_index=chunk.chunk_index,
            character_count=len(chunk.text),
            token_count=chunk.token_count,
            content_hash=chunk.content_hash,
            chunking_strategy=self._get_chunking_strategy(
                chunk
            ),
            chunking_strategy_version=(
                self._get_chunking_strategy_version(
                    chunk
                )
            ),
            provenance=provenance,
            metadata=metadata,
        )

    @staticmethod
    def _build_source_provenance(
        *,
        document: NormalizedKnowledgeDocument,
        project_id: UUID,
    ) -> KnowledgeSourceProvenance:
        source = document.source

        source_type = source.source_type

        if hasattr(source_type, "value"):
            source_type = source_type.value

        return KnowledgeSourceProvenance(
            project_id=project_id,
            source_id=source.source_id,
            source_type=str(source_type),
            source_uri=source.storage_uri,
            title=document.title,
            language=document.language,
        )

    @staticmethod
    def _build_document_provenance(
        *,
        document: NormalizedKnowledgeDocument,
        document_id: UUID,
        document_version: int,
    ) -> KnowledgeDocumentProvenance:
        return KnowledgeDocumentProvenance(
            document_id=document_id,
            document_version=document_version,
            content_hash=document.content_hash,
        )

    @staticmethod
    def _build_location_provenance(
        *,
        chunk: KnowledgeChunkDraft,
    ) -> KnowledgeLocationProvenance:
        source = chunk.source

        return KnowledgeLocationProvenance(
            element_orders=list(
                source.element_orders
            ),
            block_types=list(
                source.block_types
            ),
            page_numbers=list(
                source.page_numbers
            ),
            section_path=list(
                source.section_path
            ),
            start_time=source.start_time,
            end_time=source.end_time,
        )

    @staticmethod
    def _get_chunking_strategy(
        chunk: KnowledgeChunkDraft,
    ) -> str:
        value = chunk.metadata.get(
            "chunking_strategy"
        )

        if value is None:
            return "unknown"

        return str(value)

    @staticmethod
    def _get_chunking_strategy_version(
        chunk: KnowledgeChunkDraft,
    ) -> str:
        value = chunk.metadata.get(
            "chunking_strategy_version"
        )

        if value is None:
            value = chunk.metadata.get(
                "chunking_version"
            )

        if value is None:
            return "unknown"

        return str(value)