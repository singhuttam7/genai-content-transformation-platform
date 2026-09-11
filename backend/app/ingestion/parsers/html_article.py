from __future__ import annotations

from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True, slots=True)
class ArticleCandidate:
    """
    Represents one possible article-content container.
    """

    element: Tag
    score: float
    text: str
    paragraph_count: int
    heading_count: int
    link_density: float
    semantic_bonus: float
    boilerplate_penalty: float


@dataclass(frozen=True, slots=True)
class ArticleExtractionResult:
    """
    Result returned by HTMLArticleExtractor.

    The result contains both extracted content and diagnostics
    required for later observability and provenance.
    """

    text: str
    element: Tag
    extraction_method: str
    candidate_count: int
    selected_score: float
    metadata: dict[str, object] = field(
        default_factory=dict,
    )


class HTMLArticleExtractor:
    """
    Deterministic main-article extractor.

    Responsibilities:
    - Discover article-content candidates.
    - Score candidates using deterministic signals.
    - Penalize likely boilerplate.
    - Select the strongest candidate.
    - Provide a controlled fallback.

    This component does not:
    - perform network requests,
    - call an LLM,
    - perform metadata extraction,
    - modify the original BeautifulSoup tree.
    """

    _SEMANTIC_TAGS = {
        "article",
        "main",
    }

    _CANDIDATE_TAGS = {
        "article",
        "main",
        "section",
        "div",
    }

    _POSITIVE_CLASS_TERMS = {
        "article",
        "article-body",
        "article-content",
        "article-text",
        "content",
        "content-body",
        "content-main",
        "entry-content",
        "post-content",
        "post-body",
        "story",
        "story-body",
        "story-content",
        "main-content",
    }

    _NEGATIVE_TERMS = {
        "ad",
        "ads",
        "advert",
        "advertisement",
        "banner",
        "comment",
        "comments",
        "cookie",
        "cookie-banner",
        "cookie-consent",
        "footer",
        "header",
        "login",
        "menu",
        "navigation",
        "nav",
        "newsletter",
        "popup",
        "related",
        "related-articles",
        "search",
        "share",
        "sidebar",
        "social",
        "subscribe",
    }

    _IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "form",
    }

    _MIN_TEXT_LENGTH = 80
    _MIN_PARAGRAPHS = 1

    def extract(
        self,
        soup: BeautifulSoup,
    ) -> ArticleExtractionResult:
        """
        Extract the strongest article-content candidate.

        Raises:
            ValueError: if no meaningful article content exists.
        """

        candidates = self._discover_candidates(soup)

        scored_candidates = [
            self._score_candidate(element)
            for element in candidates
        ]

        scored_candidates = [
            candidate
            for candidate in scored_candidates
            if candidate is not None
        ]

        if scored_candidates:
            selected = max(
                scored_candidates,
                key=self._candidate_sort_key,
            )

            return ArticleExtractionResult(
                text=selected.text,
                element=selected.element,
                extraction_method=self._determine_method(
                    selected.element,
                ),
                candidate_count=len(
                    scored_candidates
                ),
                selected_score=selected.score,
                metadata=self._build_result_metadata(
                    selected,
                ),
            )

        fallback = self._fallback_extract(soup)

        if fallback is not None:
            return fallback

        raise ValueError(
            "No meaningful article content could be extracted."
        )

    def _discover_candidates(
        self,
        soup: BeautifulSoup,
    ) -> list[Tag]:
        """
        Discover possible article containers.

        Candidate discovery intentionally includes generic divs
        because many real websites do not use semantic HTML.
        """

        candidates: list[Tag] = []

        root = soup.body or soup

        for element in root.find_all(
            self._CANDIDATE_TAGS
        ):
            if not isinstance(element, Tag):
                continue

            if element.name.lower() in self._IGNORED_TAGS:
                continue

            candidates.append(element)

        return candidates

    def _score_candidate(
        self,
        element: Tag,
    ) -> ArticleCandidate | None:
        """Calculate deterministic candidate score."""

        text = self._extract_text(element)

        if len(text) < self._MIN_TEXT_LENGTH:
            return None

        paragraph_count = self._count_direct_or_nested(
            element,
            "p",
        )

        heading_count = self._count_direct_or_nested(
            element,
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            ],
        )

        link_density = self._calculate_link_density(
            element,
            text,
        )

        semantic_bonus = self._semantic_bonus(
            element,
        )

        positive_bonus = self._positive_class_bonus(
            element,
        )

        boilerplate_penalty = self._boilerplate_penalty(
            element,
        )

        paragraph_score = min(
            paragraph_count * 4.0,
            32.0,
        )

        heading_score = min(
            heading_count * 3.0,
            12.0,
        )

        text_score = min(
            len(text) / 100.0,
            40.0,
        )

        link_penalty = link_density * 30.0

        score = (
            text_score
            + paragraph_score
            + heading_score
            + semantic_bonus
            + positive_bonus
            - boilerplate_penalty
            - link_penalty
        )

        return ArticleCandidate(
            element=element,
            score=score,
            text=text,
            paragraph_count=paragraph_count,
            heading_count=heading_count,
            link_density=link_density,
            semantic_bonus=semantic_bonus,
            boilerplate_penalty=boilerplate_penalty,
        )

    @staticmethod
    def _candidate_sort_key(
        candidate: ArticleCandidate,
    ) -> tuple[float, int, int]:
        """
        Deterministic tie-breaking.

        Higher score wins.
        Then more paragraphs.
        Then more headings.
        """

        return (
            candidate.score,
            candidate.paragraph_count,
            candidate.heading_count,
        )

    def _semantic_bonus(
        self,
        element: Tag,
    ) -> float:
        """Give semantic HTML containers a strong bonus."""

        if element.name.lower() == "article":
            return 25.0

        if element.name.lower() == "main":
            return 20.0

        return 0.0

    def _positive_class_bonus(
        self,
        element: Tag,
    ) -> float:
        """Reward classes/IDs strongly associated with article content."""

        tokens = self._get_identity_tokens(element)

        score = 0.0

        for token in tokens:
            if token in self._POSITIVE_CLASS_TERMS:
                score += 10.0

        return min(
            score,
            20.0,
        )

    def _boilerplate_penalty(
        self,
        element: Tag,
    ) -> float:
        """Penalize classes and IDs associated with boilerplate."""

        tokens = self._get_identity_tokens(element)

        penalty = 0.0

        for token in tokens:
            if token in self._NEGATIVE_TERMS:
                penalty += 15.0

        return min(
            penalty,
            45.0,
        )

    @staticmethod
    def _get_identity_tokens(
        element: Tag,
    ) -> set[str]:
        """
        Extract normalized class and ID tokens.
        """

        tokens: set[str] = set()

        element_id = element.get("id")

        if isinstance(element_id, str):
            tokens.update(
                token.lower()
                for token in element_id.replace(
                    "_",
                    "-",
                ).split()
                if token
            )

        classes = element.get("class")

        if isinstance(classes, list):
            for class_name in classes:
                if not isinstance(class_name, str):
                    continue

                normalized = class_name.lower().replace(
                    "_",
                    "-",
                )

                tokens.update(
                    part
                    for part in normalized.split()
                    if part
                )

                tokens.update(
                    part
                    for part in normalized.replace(
                        "-",
                        " ",
                    ).split()
                    if part
                )

        elif isinstance(classes, str):
            normalized = classes.lower().replace(
                "_",
                "-",
            )

            tokens.update(
                part
                for part in normalized.replace(
                    "-",
                    " ",
                ).split()
                if part
            )

        return tokens

    def _calculate_link_density(
        self,
        element: Tag,
        text: str,
    ) -> float:
        """
        Calculate the proportion of text contained inside links.

        Returns a value between 0 and 1.
        """

        total_length = len(text)

        if total_length == 0:
            return 1.0

        linked_text_length = 0

        for link in element.find_all("a"):
            if not isinstance(link, Tag):
                continue

            linked_text = self._clean_text(
                link.get_text(
                    " ",
                    strip=True,
                )
            )

            linked_text_length += len(
                linked_text
            )

        return min(
            linked_text_length / total_length,
            1.0,
        )

    def _extract_text(
        self,
        element: Tag,
    ) -> str:
        """
        Extract readable text while ignoring obvious non-content tags.
        """

        clone = BeautifulSoup(
            str(element),
            "html.parser",
        )

        for tag in clone.find_all(
            self._IGNORED_TAGS
        ):
            tag.decompose()

        return self._clean_text(
            clone.get_text(
                " ",
                strip=True,
            )
        )

    @staticmethod
    def _count_direct_or_nested(
        element: Tag,
        names: str | list[str],
    ) -> int:
        """Count descendants matching one or more tag names."""

        if isinstance(names, str):
            names = [names]

        return len(
            element.find_all(names)
        )

    def _fallback_extract(
        self,
        soup: BeautifulSoup,
    ) -> ArticleExtractionResult | None:
        """
        Fallback to the document body when no strong candidate exists.

        This is intentionally conservative.
        """

        root = soup.body

        if not isinstance(root, Tag):
            return None

        text = self._extract_text(root)

        if len(text) < self._MIN_TEXT_LENGTH:
            return None

        paragraph_count = self._count_direct_or_nested(
            root,
            "p",
        )

        if paragraph_count < self._MIN_PARAGRAPHS:
            return None

        candidate = ArticleCandidate(
            element=root,
            score=0.0,
            text=text,
            paragraph_count=paragraph_count,
            heading_count=self._count_direct_or_nested(
                root,
                [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                ],
            ),
            link_density=self._calculate_link_density(
                root,
                text,
            ),
            semantic_bonus=0.0,
            boilerplate_penalty=0.0,
        )

        return ArticleExtractionResult(
            text=text,
            element=root,
            extraction_method="body_fallback",
            candidate_count=0,
            selected_score=0.0,
            metadata=self._build_result_metadata(
                candidate,
            ),
        )

    @staticmethod
    def _determine_method(
        element: Tag,
    ) -> str:
        """Return a stable extraction method identifier."""

        tag = element.name.lower()

        if tag == "article":
            return "semantic_article"

        if tag == "main":
            return "semantic_main"

        return "scored_structural_candidate"

    @staticmethod
    def _build_result_metadata(
        candidate: ArticleCandidate,
    ) -> dict[str, object]:
        """Build diagnostics for observability/provenance."""

        return {
            "paragraph_count": candidate.paragraph_count,
            "heading_count": candidate.heading_count,
            "link_density": round(
                candidate.link_density,
                6,
            ),
            "semantic_bonus": candidate.semantic_bonus,
            "boilerplate_penalty": candidate.boilerplate_penalty,
        }

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """Normalize extracted article text."""

        text = text.replace(
            "\xa0",
            " ",
        )

        text = " ".join(
            text.split()
        ).strip()

        punctuation = ",.!?;:%)]}"

        for mark in punctuation:
            text = text.replace(
                f" {mark}",
                mark,
            )

        opening_punctuation = "([{"

        for mark in opening_punctuation:
            text = text.replace(
                f"{mark} ",
                mark,
            )

        return text