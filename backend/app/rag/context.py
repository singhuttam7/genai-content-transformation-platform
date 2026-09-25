from __future__ import annotations

from typing import Any

from app.rag.config import RAGContextConfig
from app.rag.schemas import (
    RAGContext,
    RAGQuery,
    RAGRetrievedChunk,
)


class RAGContextAssembler:
    """
    Assemble retrieved RAG chunks into deterministic bounded context.

    Responsibilities:
    - validate the supplied RAG query and chunks;
    - apply deterministic context limits;
    - preserve retrieval ordering;
    - construct deterministic context text;
    - preserve chunk objects without mutating caller-owned data;
    - expose context-limit metadata.

    Non-responsibilities:
    - retrieval;
    - embedding;
    - ranking;
    - token counting;
    - prompt construction;
    - LLM invocation.
    """

    CONTEXT_HEADER = "Retrieved Knowledge Context"

    def __init__(
        self,
        config: RAGContextConfig | None = None,
    ) -> None:
        self._config = config or RAGContextConfig()

    @property
    def config(self) -> RAGContextConfig:
        """Return the immutable context configuration."""
        return self._config

    def assemble(
        self,
        *,
        query: RAGQuery,
        chunks: list[RAGRetrievedChunk],
    ) -> RAGContext:
        """
        Assemble retrieved chunks into deterministic bounded context.

        Retrieval order is preserved exactly. Chunks are selected from
        the beginning of the supplied retrieval result until either the
        chunk-count or character budget is reached.
        """

        self._validate_query(query)
        self._validate_chunks(chunks)

        selected_chunks, truncation_count = self._select_chunks(
            chunks,
        )

        context_text = self._build_context_text(
            selected_chunks,
        )

        metadata = self._build_metadata(
            selected_chunks,
            source_chunk_count=len(chunks),
            truncation_count=truncation_count,
        )

        return RAGContext(
            query=query,
            chunks=selected_chunks,
            context_text=context_text,
            metadata=metadata,
        )

    def _select_chunks(
        self,
        chunks: list[RAGRetrievedChunk],
    ) -> tuple[list[RAGRetrievedChunk], int]:
        """
        Select chunks deterministically under configured limits.

        The final rendered context is always checked against the actual
        character budget. If a chunk does not fit completely, its text
        is deterministically truncated to the largest size that fits.
        """

        selected: list[RAGRetrievedChunk] = []
        truncation_count = 0

        for chunk in chunks:
            if len(selected) >= self._config.max_chunks:
                break

            candidate = selected + [
                chunk.model_copy(deep=True),
            ]

            candidate_text = self._build_context_text(
                candidate,
            )

            if (
                len(candidate_text)
                <= self._config.max_context_characters
            ):
                selected.append(candidate[-1])
                continue

            current_text = self._build_context_text(
                selected,
            )

            remaining = (
                self._config.max_context_characters
                - len(current_text)
            )

            if selected:
                separator = "\n\n"
                prefix = self._chunk_prefix(
                    len(selected) + 1,
                )
                fixed_overhead = len(separator) + len(prefix) + 1
            else:
                fixed_overhead = (
                    len(self.CONTEXT_HEADER)
                    + 2
                    + len(self._chunk_prefix(1))
                    + 1
                )

            available_text = remaining - fixed_overhead

            if available_text <= 0:
                break

            truncated_text = chunk.text[
                :available_text
            ].rstrip()

            if not truncated_text:
                break

            truncated_chunk = chunk.model_copy(
                update={
                    "text": truncated_text,
                },
                deep=True,
            )

            final_candidate = selected + [
                truncated_chunk,
            ]

            final_text = self._build_context_text(
                final_candidate,
            )

            if (
                len(final_text)
                > self._config.max_context_characters
            ):
                # Defensive correction for whitespace removal or any
                # formatting difference in the final representation.
                excess = (
                    len(final_text)
                    - self._config.max_context_characters
                )

                final_text_length = max(
                    0,
                    len(truncated_chunk.text) - excess,
                )

                truncated_chunk = chunk.model_copy(
                    update={
                        "text": truncated_chunk.text[
                            :final_text_length
                        ].rstrip(),
                    },
                    deep=True,
                )

                final_candidate = selected + [
                    truncated_chunk,
                ]

                final_text = self._build_context_text(
                    final_candidate,
                )

            if not final_text:
                break

            if (
                len(final_text)
                <= self._config.max_context_characters
            ):
                selected.append(truncated_chunk)
                truncation_count += 1

            break

        return selected, truncation_count

    @classmethod
    def _build_context_text(
        cls,
        chunks: list[RAGRetrievedChunk],
    ) -> str:
        """
        Build deterministic human-readable retrieval context.
        """

        if not chunks:
            return ""

        sections = [cls.CONTEXT_HEADER]

        for index, chunk in enumerate(chunks, start=1):
            sections.extend(
                [
                    "",
                    cls._chunk_prefix(index),
                    chunk.text,
                ]
            )

        return "\n".join(sections)

    @staticmethod
    def _chunk_prefix(index: int) -> str:
        return f"[Chunk {index}]"

    def _build_metadata(
        self,
        chunks: list[RAGRetrievedChunk],
        *,
        source_chunk_count: int,
        truncation_count: int,
    ) -> dict[str, Any]:
        """Build deterministic aggregate metadata."""

        similarities = [
            chunk.similarity
            for chunk in chunks
        ]

        return {
            "chunk_count": len(chunks),
            "source_chunk_count": source_chunk_count,
            "chunk_ids": [
                str(chunk.chunk_id)
                for chunk in chunks
            ],
            "similarities": similarities,
            "has_provenance": any(
                chunk.provenance is not None
                for chunk in chunks
            ),
            "max_chunks": self._config.max_chunks,
            "max_context_characters": (
                self._config.max_context_characters
            ),
            "truncated_chunk_count": truncation_count,
        }

    @staticmethod
    def _validate_query(
        query: RAGQuery,
    ) -> None:
        if not isinstance(query, RAGQuery):
            raise TypeError(
                "query must be a RAGQuery."
            )

    @staticmethod
    def _validate_chunks(
        chunks: list[RAGRetrievedChunk],
    ) -> None:
        if not isinstance(chunks, list):
            raise TypeError(
                "chunks must be a list of RAGRetrievedChunk."
            )

        for chunk in chunks:
            if not isinstance(
                chunk,
                RAGRetrievedChunk,
            ):
                raise TypeError(
                    "chunks must contain only RAGRetrievedChunk objects."
                )