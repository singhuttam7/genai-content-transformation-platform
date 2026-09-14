from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from app.ingestion.schemas import ContentBlockType
from app.knowledge.chunking.config import ChunkingConfig
from app.knowledge.chunking.schemas import (
    ChunkSourceReference,
    ChunkingResult,
    ChunkingStatistics,
    KnowledgeChunkDraft,
)
from app.knowledge.normalization.schemas import (
    NormalizedKnowledgeDocument,
    NormalizedKnowledgeElement,
)


@dataclass
class _ChunkBuffer:
    """Internal mutable representation of a chunk."""

    text: str
    element_orders: list[int]
    block_types: list[ContentBlockType]
    page_numbers: list[int]
    start_time: float | None
    end_time: float | None
    section_path: list[str]


class StructureAwareChunker:
    """
    Deterministic structure-aware chunker.

    Responsibilities:
    - preserve document structure
    - preserve heading context
    - preserve source provenance
    - preserve page/timestamp information
    - split oversized content deterministically
    - optionally apply bounded overlap

    The chunker intentionally does not depend on:
    - LLMs
    - embeddings
    - LangChain
    - LangGraph
    - vector databases
    """

    STRATEGY = "structure_aware"
    VERSION = "1.0"

    HEADING_TYPES = frozenset(
        {
            ContentBlockType.HEADING,
        }
    )

    SENTENCE_BOUNDARY_PATTERN = re.compile(
        r"(?<=[.!?।！？])\s+"
    )

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()

    async def chunk(
        self,
        document: NormalizedKnowledgeDocument,
    ) -> ChunkingResult:
        """Convert normalized content into deterministic knowledge chunks."""

        buffers: list[_ChunkBuffer] = []

        current: _ChunkBuffer | None = None
        section_path: list[str] = []

        for element in document.elements:
            if not self._should_include(element):
                continue

            if self._is_heading(element):
                section_path = self._update_section_path(
                    section_path=section_path,
                    heading=element.content,
                )

                if current is not None and current.text.strip():
                    buffers.extend(
                        self._split_buffer(current)
                    )
                    current = None

                if self.config.preserve_headings:
                    current = self._create_buffer(
                        element=element,
                        section_path=section_path,
                    )

                continue

            if not element.content.strip():
                continue

            if current is None:
                current = self._create_buffer(
                    element=element,
                    section_path=section_path,
                )
                continue

            candidate = self._append_element(
                current=current,
                element=element,
                section_path=section_path,
            )

            if len(candidate.text) <= self.config.max_characters:
                current = candidate
                continue

            if current.text.strip():
                buffers.extend(
                    self._split_buffer(current)
                )

            current = self._create_buffer(
                element=element,
                section_path=section_path,
            )

        if current is not None and current.text.strip():
            buffers.extend(
                self._split_buffer(current)
            )

        return self._build_result(
            document=document,
            buffers=buffers,
        )

    def _should_include(
        self,
        element: NormalizedKnowledgeElement,
    ) -> bool:
        """Determine whether an element participates in chunking."""

        if element.content.strip():
            return True

        return (
            self.config.include_non_text_blocks
            and element.block_type not in self.HEADING_TYPES
        )

    def _is_heading(
        self,
        element: NormalizedKnowledgeElement,
    ) -> bool:
        return element.block_type in self.HEADING_TYPES

    @staticmethod
    def _update_section_path(
        *,
        section_path: list[str],
        heading: str,
    ) -> list[str]:
        """
        Update the current section path.

        A4 does not expose heading depth, so the current implementation
        maintains one active heading.
        """

        normalized_heading = heading.strip()

        if not normalized_heading:
            return list(section_path)

        return [normalized_heading]

    def _create_buffer(
        self,
        *,
        element: NormalizedKnowledgeElement,
        section_path: list[str],
    ) -> _ChunkBuffer:
        return _ChunkBuffer(
            text=element.content.strip(),
            element_orders=[
                self._canonical_element_order(element)
            ],
            block_types=[
                element.block_type
            ],
            page_numbers=(
                [element.page_number]
                if element.page_number is not None
                else []
            ),
            start_time=element.start_time,
            end_time=element.end_time,
            section_path=list(section_path),
        )

    def _append_element(
        self,
        *,
        current: _ChunkBuffer,
        element: NormalizedKnowledgeElement,
        section_path: list[str],
    ) -> _ChunkBuffer:
        """Append an element while preserving provenance."""

        new_text = self._join_text(
            current.text,
            element.content.strip(),
        )

        page_numbers = list(current.page_numbers)

        if (
            element.page_number is not None
            and element.page_number not in page_numbers
        ):
            page_numbers.append(element.page_number)

        start_time = current.start_time

        if start_time is None:
            start_time = element.start_time

        end_time = element.end_time

        if end_time is None:
            end_time = current.end_time

        return _ChunkBuffer(
            text=new_text,
            element_orders=[
                *current.element_orders,
                self._canonical_element_order(element),
            ],
            block_types=[
                *current.block_types,
                element.block_type,
            ],
            page_numbers=page_numbers,
            start_time=start_time,
            end_time=end_time,
            section_path=(
                list(section_path)
                if section_path
                else list(current.section_path)
            ),
        )

    @staticmethod
    def _canonical_element_order(
        element: NormalizedKnowledgeElement,
    ) -> int:
        return element.order

    @staticmethod
    def _join_text(
        first: str,
        second: str,
    ) -> str:
        first = first.strip()
        second = second.strip()

        if not first:
            return second

        if not second:
            return first

        return f"{first}\n\n{second}"

    def _split_buffer(
        self,
        buffer: _ChunkBuffer,
    ) -> list[_ChunkBuffer]:
        """
        Split an oversized buffer.

        Overlap is applied only to semantic word/sentence/paragraph pieces.
        Hard character splits remain exact and non-overlapping.
        """

        if len(buffer.text) <= self.config.max_characters:
            return [buffer]

        pieces, allow_overlap = self._split_text(
            buffer.text,
            max_characters=self.config.max_characters,
        )

        if not pieces:
            return []

        if allow_overlap:
            pieces = self._apply_overlap(
                pieces,
                max_characters=self.config.max_characters,
                overlap_characters=self.config.overlap_characters,
            )

        return [
            _ChunkBuffer(
                text=piece,
                element_orders=list(buffer.element_orders),
                block_types=list(buffer.block_types),
                page_numbers=list(buffer.page_numbers),
                start_time=buffer.start_time,
                end_time=buffer.end_time,
                section_path=list(buffer.section_path),
            )
            for piece in pieces
        ]

    def _split_text(
        self,
        text: str,
        *,
        max_characters: int,
    ) -> tuple[list[str], bool]:
        """
        Hierarchical deterministic splitting.

        Returns:
            (pieces, allow_overlap)

        Hard character splitting returns allow_overlap=False because
        exact character partitioning is part of the existing contract.
        """

        text = text.strip()

        if not text:
            return [], False

        if len(text) <= max_characters:
            return [text], False

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

        if len(paragraphs) > 1:
            pieces: list[str] = []
            current = ""
            allow_overlap = True

            for paragraph in paragraphs:
                candidate = self._join_text(
                    current,
                    paragraph,
                )

                if len(candidate) <= max_characters:
                    current = candidate
                    continue

                if current:
                    pieces.append(current)
                    current = ""

                if len(paragraph) <= max_characters:
                    current = paragraph
                else:
                    paragraph_pieces, paragraph_overlap = (
                        self._split_sentences(
                            paragraph,
                            max_characters=max_characters,
                        )
                    )

                    pieces.extend(paragraph_pieces)
                    allow_overlap = (
                        allow_overlap and paragraph_overlap
                    )

            if current:
                pieces.append(current)

            return pieces, allow_overlap

        return self._split_sentences(
            text,
            max_characters=max_characters,
        )

    def _split_sentences(
        self,
        text: str,
        *,
        max_characters: int,
    ) -> tuple[list[str], bool]:
        sentences = [
            sentence.strip()
            for sentence in self.SENTENCE_BOUNDARY_PATTERN.split(text)
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            return self._split_words(
                text,
                max_characters=max_characters,
            )

        pieces: list[str] = []
        current = ""
        allow_overlap = True

        for sentence in sentences:
            if len(sentence) > max_characters:
                if current:
                    pieces.append(current)
                    current = ""

                sentence_pieces, sentence_overlap = (
                    self._split_words(
                        sentence,
                        max_characters=max_characters,
                    )
                )

                pieces.extend(sentence_pieces)
                allow_overlap = (
                    allow_overlap and sentence_overlap
                )
                continue

            candidate = self._join_text(
                current,
                sentence,
            )

            if len(candidate) <= max_characters:
                current = candidate
                continue

            if current:
                pieces.append(current)

            current = sentence

        if current:
            pieces.append(current)

        return pieces, allow_overlap

    def _split_words(
        self,
        text: str,
        *,
        max_characters: int,
    ) -> tuple[list[str], bool]:
        words = text.split()

        if not words:
            return [], False

        pieces: list[str] = []
        current = ""

        for word in words:
            if len(word) > max_characters:
                if current:
                    pieces.append(current)
                    current = ""

                pieces.extend(
                    self._hard_split(
                        word,
                        max_characters=max_characters,
                    )
                )

                # Hard split means exact partitioning.
                return pieces, False

            candidate = (
                word
                if not current
                else f"{current} {word}"
            )

            if len(candidate) <= max_characters:
                current = candidate
                continue

            if current:
                pieces.append(current)

            current = word

        if current:
            pieces.append(current)

        return pieces, True

    @staticmethod
    def _hard_split(
        text: str,
        *,
        max_characters: int,
    ) -> list[str]:
        """Split a long token into exact character-sized pieces."""

        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero."
            )

        return [
            text[index : index + max_characters]
            for index in range(
                0,
                len(text),
                max_characters,
            )
        ]

    def _apply_overlap(
        self,
        pieces: list[str],
        *,
        max_characters: int,
        overlap_characters: int,
    ) -> list[str]:
        """
        Apply bounded overlap between adjacent semantic pieces.

        The final result always satisfies:

            len(piece) <= max_characters
        """

        if (
            overlap_characters <= 0
            or len(pieces) <= 1
        ):
            return list(pieces)

        result: list[str] = [
            pieces[0][:max_characters]
        ]

        for piece in pieces[1:]:
            piece = piece.strip()

            if not piece:
                continue

            # One character is required for the separator.
            available = max_characters - len(piece)

            if available <= 1:
                result.append(
                    piece[:max_characters]
                )
                continue

            requested_overlap = min(
                overlap_characters,
                available - 1,
            )

            overlap_text = self._get_overlap_text(
                result[-1],
                requested_overlap,
            )

            if not overlap_text:
                result.append(
                    piece[:max_characters]
                )
                continue

            candidate = (
                f"{overlap_text} {piece}"
            ).strip()

            if len(candidate) > max_characters:
                allowed_overlap = max(
                    0,
                    max_characters - len(piece) - 1,
                )

                overlap_text = self._trim_overlap(
                    overlap_text,
                    max_characters=allowed_overlap,
                )

                if overlap_text:
                    candidate = (
                        f"{overlap_text} {piece}"
                    ).strip()
                else:
                    candidate = piece

            # Absolute invariant.
            if len(candidate) > max_characters:
                candidate = candidate[
                    :max_characters
                ].rstrip()

            result.append(candidate)

        return result

    @staticmethod
    def _get_overlap_text(
        text: str,
        overlap_characters: int,
    ) -> str:
        """Extract a word-boundary-aware suffix."""

        if (
            not text
            or overlap_characters <= 0
        ):
            return ""

        text = text.strip()

        if len(text) <= overlap_characters:
            return text

        words = text.split()

        selected: list[str] = []
        length = 0

        for word in reversed(words):
            additional_length = len(word)

            if selected:
                additional_length += 1

            if length + additional_length > overlap_characters:
                break

            selected.insert(0, word)
            length += additional_length

        if selected:
            return " ".join(selected)

        return text[-overlap_characters:]

    @staticmethod
    def _trim_overlap(
        overlap_text: str,
        *,
        max_characters: int,
    ) -> str:
        """Trim overlap while preserving whole-word boundaries."""

        if max_characters <= 0:
            return ""

        overlap_text = overlap_text.strip()

        if len(overlap_text) <= max_characters:
            return overlap_text

        words = overlap_text.split()

        selected: list[str] = []
        length = 0

        for word in reversed(words):
            additional_length = len(word)

            if selected:
                additional_length += 1

            if length + additional_length > max_characters:
                break

            selected.insert(0, word)
            length += additional_length

        if selected:
            return " ".join(selected)

        return overlap_text[-max_characters:]

    def _buffer_to_chunk(
        self,
        *,
        buffer: _ChunkBuffer,
        chunk_index: int,
    ) -> KnowledgeChunkDraft:
        text = buffer.text.strip()

        content_hash = self._calculate_chunk_hash(
            text=text,
            source=buffer,
        )

        source = ChunkSourceReference(
            element_orders=sorted(
                set(buffer.element_orders)
            ),
            block_types=list(
                dict.fromkeys(
                    buffer.block_types
                )
            ),
            page_numbers=sorted(
                set(buffer.page_numbers)
            ),
            start_time=buffer.start_time,
            end_time=buffer.end_time,
            section_path=list(
                buffer.section_path
            ),
        )

        metadata = {
            "chunking_strategy": self.STRATEGY,
            "chunking_strategy_version": self.VERSION,
            "chunking_version": self.VERSION,
            "overlap_characters": (
                self.config.overlap_characters
            ),
            "section_path": list(
                buffer.section_path
            ),
        }

        return KnowledgeChunkDraft(
            chunk_index=chunk_index,
            text=text,
            content_hash=content_hash,
            token_count=None,
            source=source,
            metadata=metadata,
        )

    @staticmethod
    def _calculate_chunk_hash(
        *,
        text: str,
        source: _ChunkBuffer,
    ) -> str:
        hasher = hashlib.sha256()

        hasher.update(
            text.encode("utf-8")
        )

        for order in source.element_orders:
            hasher.update(b"\x00order:")
            hasher.update(
                str(order).encode("utf-8")
            )

        for block_type in source.block_types:
            hasher.update(b"\x00type:")
            hasher.update(
                block_type.value.encode("utf-8")
            )

        for page_number in source.page_numbers:
            hasher.update(b"\x00page:")
            hasher.update(
                str(page_number).encode("utf-8")
            )

        if source.start_time is not None:
            hasher.update(b"\x00start:")
            hasher.update(
                str(source.start_time).encode("utf-8")
            )

        if source.end_time is not None:
            hasher.update(b"\x00end:")
            hasher.update(
                str(source.end_time).encode("utf-8")
            )

        for section in source.section_path:
            hasher.update(b"\x00section:")
            hasher.update(
                section.encode("utf-8")
            )

        return hasher.hexdigest()

    def _empty_result(
        self,
        *,
        document: NormalizedKnowledgeDocument,
    ) -> ChunkingResult:
        return ChunkingResult(
            chunks=[],
            statistics=ChunkingStatistics(
                input_element_count=len(
                    document.elements
                ),
                output_chunk_count=0,
                input_character_count=len(
                    document.text
                ),
                output_character_count=0,
                largest_chunk_characters=0,
                smallest_chunk_characters=0,
            ),
            strategy=self.STRATEGY,
            strategy_version=self.VERSION,
        )

    def _build_result(
        self,
        *,
        document: NormalizedKnowledgeDocument,
        buffers: Iterable[_ChunkBuffer],
    ) -> ChunkingResult:
        valid_buffers = [
            buffer
            for buffer in buffers
            if buffer.text.strip()
        ]

        if not valid_buffers:
            return self._empty_result(
                document=document,
            )

        chunks = [
            self._buffer_to_chunk(
                buffer=buffer,
                chunk_index=index,
            )
            for index, buffer in enumerate(
                valid_buffers
            )
        ]

        lengths = [
            len(chunk.text)
            for chunk in chunks
        ]

        return ChunkingResult(
            chunks=chunks,
            statistics=ChunkingStatistics(
                input_element_count=len(
                    document.elements
                ),
                output_chunk_count=len(chunks),
                input_character_count=len(
                    document.text
                ),
                output_character_count=sum(
                    lengths
                ),
                largest_chunk_characters=max(
                    lengths
                ),
                smallest_chunk_characters=min(
                    lengths
                ),
            ),
            strategy=self.STRATEGY,
            strategy_version=self.VERSION,
        )