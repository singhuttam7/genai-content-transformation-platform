from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from app.ingestion.parsers.html_article import HTMLArticleExtractor
from app.ingestion.parsers.html_metadata import HTMLMetadataExtractor
from app.ingestion.processor import ContentProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)


class HTMLDocumentProcessor(ContentProcessor):
    """
    Deterministic HTML document processor.

    Responsibilities:
    - Parse HTML content.
    - Remove non-content elements.
    - Extract document metadata.
    - Attempt main-article extraction.
    - Fall back to structural HTML extraction when article
      extraction is not applicable.
    - Preserve structural blocks such as headings, paragraphs,
      lists, quotes, code, tables, and figures.

    Network fetching is intentionally handled outside this processor.
    """

    supported_types = (
        InputType.URL,
    )

    _REMOVED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "iframe",
        "svg",
        "canvas",
        "form",
    }

    _HEADING_TAGS = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    _STRUCTURAL_CONTAINER_TAGS = {
        "ul",
        "ol",
        "table",
        "figure",
        "blockquote",
        "pre",
    }

    def __init__(
        self,
        metadata_extractor: HTMLMetadataExtractor | None = None,
        article_extractor: HTMLArticleExtractor | None = None,
    ) -> None:
        self.metadata_extractor = (
            metadata_extractor
            or HTMLMetadataExtractor()
        )

        self.article_extractor = (
            article_extractor
            or HTMLArticleExtractor()
        )

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        """
        Process HTML content into ExtractedContent.

        The request content must already be resolved by the
        ingestion/fetching layer.
        """

        if request.input_type != InputType.URL:
            raise ValueError(
                "HTMLDocumentProcessor only supports URL input."
            )

        if request.content is None:
            raise ValueError(
                "HTML content must be resolved before processing."
            )

        if isinstance(request.content, str):
            content = request.content.encode("utf-8")
        else:
            content = request.content

        return self.parse(
            content,
            request=request,
            metadata_extractor=self.metadata_extractor,
            article_extractor=self.article_extractor,
        )

    @classmethod
    def parse(
        cls,
        content: bytes,
        *,
        request: IngestionRequest,
        metadata_extractor: HTMLMetadataExtractor | None = None,
        article_extractor: HTMLArticleExtractor | None = None,
    ) -> ExtractedContent:
        """
        Parse HTML bytes without performing network access.
        """

        if not content:
            raise ValueError(
                "HTML content cannot be empty."
            )

        html = content.decode(
            "utf-8",
            errors="replace",
        )

        if not html.strip():
            raise ValueError(
                "HTML content cannot be empty."
            )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        cls._remove_non_content_elements(soup)

        metadata_extractor = (
            metadata_extractor
            or HTMLMetadataExtractor()
        )

        article_extractor = (
            article_extractor
            or HTMLArticleExtractor()
        )

        # ---------------------------------------------------------
        # Metadata extraction
        # ---------------------------------------------------------

        metadata = metadata_extractor.extract(
            soup,
            source_url=request.url,
            request_title=request.title,
        )

        # ---------------------------------------------------------
        # Article extraction with structural fallback
        # ---------------------------------------------------------

        article_blocks: list[ContentBlock]

        try:
            article_result = article_extractor.extract(
                soup,
            )

            article_blocks = cls._build_blocks_from_element(
                article_result.element,
            )

            if not article_blocks:
                raise ValueError(
                    "Article extraction produced no structural blocks."
                )

            article_metadata = {
                "method": article_result.extraction_method,
                "candidate_count": article_result.candidate_count,
                "selected_score": article_result.selected_score,
                **article_result.metadata,
            }

        except ValueError:
            # Article extraction is an enhancement.
            #
            # Small HTML documents, simple test documents,
            # fragments, lists, tables, figures, etc. may not
            # satisfy article extraction requirements.
            #
            # Preserve the original structural HTML behaviour
            # in those cases.

            fallback_root = soup.body or soup

            article_blocks = cls._build_blocks_from_element(
                fallback_root,
            )

            article_metadata = {
                "method": "structural_fallback",
                "candidate_count": 0,
                "selected_score": None,
            }

        # ---------------------------------------------------------
        # Final content validation
        # ---------------------------------------------------------

        if not article_blocks:
            raise ValueError(
                "No meaningful content could be extracted."
            )

        text = cls._build_text(
            article_blocks,
        )

        if not text:
            raise ValueError(
                "No meaningful content could be extracted."
            )

        # ---------------------------------------------------------
        # Merge metadata
        # ---------------------------------------------------------

        merged_metadata = dict(
            request.metadata
        )

        merged_metadata.update(
            metadata
        )

        merged_metadata[
            "article_extraction"
        ] = article_metadata

        # ---------------------------------------------------------
        # Source information
        # ---------------------------------------------------------

        title = (
            str(metadata["title"])
            if metadata.get("title")
            else request.title
        )

        source = SourceReference(
            source_id=request.source_id,
            source_type=InputType.URL,
            title=title,
            filename=request.filename,
            mime_type=(
                request.mime_type
                or "text/html"
            ),
            storage_uri=request.storage_uri,
        )

        # ---------------------------------------------------------
        # Final ExtractedContent
        # ---------------------------------------------------------

        return ExtractedContent(
            source=source,
            title=title,
            text=text,
            blocks=article_blocks,
            language=(
                str(metadata["language"])
                if metadata.get("language")
                else None
            ),
            metadata=merged_metadata,
        )

    # =============================================================
    # HTML CLEANING
    # =============================================================

    @classmethod
    def _remove_non_content_elements(
        cls,
        soup: BeautifulSoup,
    ) -> None:
        """
        Remove HTML elements that cannot provide useful
        document content.
        """

        for element in soup.find_all(
            cls._REMOVED_TAGS
        ):
            element.decompose()

    # =============================================================
    # STRUCTURAL BLOCK EXTRACTION
    # =============================================================

    @classmethod
    def _build_blocks_from_element(
        cls,
        root: Tag | BeautifulSoup,
    ) -> list[ContentBlock]:
        """
        Extract structural content blocks from a selected HTML root.

        Supported structures:
        - headings
        - paragraphs
        - blockquotes
        - lists
        - code blocks
        - tables
        - figures

        Structural containers such as tables and lists are treated
        as atomic blocks to avoid duplicate descendant extraction.
        """

        blocks: list[ContentBlock] = []

        elements = root.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "blockquote",
                "ul",
                "ol",
                "pre",
                "table",
                "figure",
            ]
        )

        # If the root itself is a structural element, include it.
        if isinstance(root, Tag):
            root_tag = root.name.lower()

            if root_tag in (
                cls._HEADING_TAGS
                | {
                    "p",
                    "blockquote",
                    "ul",
                    "ol",
                    "pre",
                    "table",
                    "figure",
                }
            ):
                elements.insert(
                    0,
                    root,
                )

        for element in elements:
            if not isinstance(
                element,
                Tag,
            ):
                continue

            # Prevent descendants of atomic structural containers
            # from becoming duplicate blocks.
            if (
                element is not root
                and cls._has_structural_parent(
                    element,
                    boundary=root,
                )
            ):
                continue

            block = cls._element_to_block(
                element,
                order=len(blocks),
            )

            if block is not None:
                blocks.append(block)

        return blocks

    @classmethod
    def _element_to_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        """
        Convert one structural HTML element into a ContentBlock.
        """

        tag = element.name.lower()

        if tag in cls._HEADING_TAGS:
            return cls._heading_block(
                element,
                order=order,
            )

        if tag == "p":
            return cls._paragraph_block(
                element,
                order=order,
            )

        if tag == "blockquote":
            return cls._quote_block(
                element,
                order=order,
            )

        if tag in {
            "ul",
            "ol",
        }:
            return cls._list_block(
                element,
                order=order,
            )

        if tag == "pre":
            return cls._code_block(
                element,
                order=order,
            )

        if tag == "table":
            return cls._table_block(
                element,
                order=order,
            )

        if tag == "figure":
            return cls._figure_block(
                element,
                order=order,
            )

        return None

    # =============================================================
    # HEADING
    # =============================================================

    @classmethod
    def _heading_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        content = cls._clean_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not content:
            return None

        return ContentBlock(
            block_type=ContentBlockType.HEADING,
            content=content,
            order=order,
            metadata={
                "html_tag": element.name.lower(),
            },
        )

    # =============================================================
    # PARAGRAPH
    # =============================================================

    @classmethod
    def _paragraph_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        content = cls._clean_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not content:
            return None

        return ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content=content,
            order=order,
            metadata={
                "html_tag": "p",
            },
        )

    # =============================================================
    # BLOCKQUOTE
    # =============================================================

    @classmethod
    def _quote_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        content = cls._clean_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not content:
            return None

        return ContentBlock(
            block_type=ContentBlockType.QUOTE,
            content=content,
            order=order,
            metadata={
                "html_tag": "blockquote",
            },
        )

    # =============================================================
    # LIST
    # =============================================================

    @classmethod
    def _list_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        items: list[str] = []

        # Only direct <li> children are processed.
        #
        # This prevents nested lists from becoming separate
        # structural blocks.
        for item in element.find_all(
            "li",
            recursive=False,
        ):
            text = cls._clean_text(
                item.get_text(
                    " ",
                    strip=True,
                )
            )

            if text:
                items.append(text)

        if not items:
            return None

        return ContentBlock(
            block_type=ContentBlockType.LIST,
            content="\n".join(items),
            order=order,
            metadata={
                "html_tag": element.name.lower(),
                "ordered": (
                    element.name.lower() == "ol"
                ),
            },
        )

    # =============================================================
    # CODE
    # =============================================================

    @classmethod
    def _code_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        content = element.get_text(
            "\n",
            strip=False,
        ).strip()

        if not content:
            return None

        return ContentBlock(
            block_type=ContentBlockType.CODE,
            content=content,
            order=order,
            metadata={
                "html_tag": "pre",
            },
        )

    # =============================================================
    # TABLE
    # =============================================================

    @classmethod
    def _table_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        rows: list[str] = []

        for row in element.find_all(
            "tr",
            recursive=True,
        ):
            cells: list[str] = []

            for cell in row.find_all(
                [
                    "th",
                    "td",
                ],
                recursive=False,
            ):
                text = cls._clean_text(
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                )

                cells.append(text)

            if cells:
                rows.append(
                    " | ".join(cells)
                )

        if not rows:
            return None

        return ContentBlock(
            block_type=ContentBlockType.TABLE,
            content="\n".join(rows),
            order=order,
            metadata={
                "html_tag": "table",
                "rows": len(rows),
            },
        )

    # =============================================================
    # FIGURE / IMAGE
    # =============================================================

    @classmethod
    def _figure_block(
        cls,
        element: Tag,
        *,
        order: int,
    ) -> ContentBlock | None:
        image = element.find(
            "img"
        )

        caption = element.find(
            "figcaption"
        )

        metadata: dict[str, object] = {
            "html_tag": "figure",
        }

        parts: list[str] = []

        # ---------------------------------------------------------
        # Image
        # ---------------------------------------------------------

        if isinstance(
            image,
            Tag,
        ):
            alt = image.get(
                "alt"
            )

            if (
                isinstance(
                    alt,
                    str,
                )
                and alt.strip()
            ):
                metadata["alt"] = alt.strip()

                parts.append(
                    alt.strip()
                )

            src = image.get(
                "src"
            )

            if (
                isinstance(
                    src,
                    str,
                )
                and src.strip()
            ):
                metadata["src"] = src.strip()

        # ---------------------------------------------------------
        # Caption
        # ---------------------------------------------------------

        if isinstance(
            caption,
            Tag,
        ):
            caption_text = cls._clean_text(
                caption.get_text(
                    " ",
                    strip=True,
                )
            )

            if caption_text:
                metadata[
                    "caption"
                ] = caption_text

                parts.append(
                    caption_text
                )

        if not parts:
            return None

        return ContentBlock(
            block_type=ContentBlockType.IMAGE,
            content="\n".join(parts),
            order=order,
            metadata=metadata,
        )

    # =============================================================
    # STRUCTURAL DUPLICATE PREVENTION
    # =============================================================

    @classmethod
    def _has_structural_parent(
        cls,
        element: Tag,
        *,
        boundary: Tag | BeautifulSoup,
    ) -> bool:
        """
        Determine whether an element is nested inside an atomic
        structural container.

        Example:

            <ul>
                <li>One</li>
                <li>Two</li>
            </ul>

        The <li> elements should not become independent blocks.

        The search stops when the selected article/fallback root
        is reached.
        """

        parent = element.parent

        while isinstance(
            parent,
            Tag,
        ):
            if parent is boundary:
                return False

            if (
                parent.name.lower()
                in cls._STRUCTURAL_CONTAINER_TAGS
            ):
                return True

            parent = parent.parent

        return False

    # =============================================================
    # TEXT NORMALIZATION
    # =============================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Normalize readable inline text.
        """

        # Convert non-breaking spaces.
        text = text.replace(
            "\xa0",
            " ",
        )

        # Normalize whitespace.
        text = " ".join(
            text.split()
        ).strip()

        # Remove whitespace before punctuation.
        punctuation = ",.!?;:%)]}"

        for mark in punctuation:
            text = text.replace(
                f" {mark}",
                mark,
            )

        # Remove whitespace after opening punctuation.
        opening_punctuation = "([{"

        for mark in opening_punctuation:
            text = text.replace(
                f"{mark} ",
                mark,
            )

        return text

    # =============================================================
    # DOCUMENT TEXT
    # =============================================================

    @staticmethod
    def _build_text(
        blocks: list[ContentBlock],
    ) -> str:
        """
        Build normalized document text from structural blocks.
        """

        return "\n\n".join(
            block.content
            for block in blocks
            if block.content.strip()
        ).strip()