from __future__ import annotations

from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag


class HTMLMetadataExtractor:
    """
    Deterministic metadata extractor for HTML documents.

    This component only interprets already-parsed HTML.
    It does not perform network requests or article-content extraction.

    Supported metadata:
    - HTML title
    - Meta description
    - Author
    - Language
    - Canonical URL
    - Keywords
    - OpenGraph metadata
    - Article metadata
    """

    _OPEN_GRAPH_FIELDS = {
        "og:title": "title",
        "og:description": "description",
        "og:type": "type",
        "og:site_name": "site_name",
        "og:image": "image",
        "og:url": "url",
        "og:locale": "locale",
    }

    _ARTICLE_FIELDS = {
        "article:section": "section",
        "article:published_time": "published_time",
        "article:modified_time": "modified_time",
        "article:author": "author",
    }

    _DESCRIPTION_NAMES = (
        "description",
    )

    _AUTHOR_NAMES = (
        "author",
    )

    _KEYWORD_NAMES = (
        "keywords",
    )

    _LANGUAGE_META_NAMES = (
        "language",
        "content-language",
    )

    _TITLE_FALLBACK_FIELDS = (
        "og:title",
    )

    def extract(
        self,
        soup: BeautifulSoup,
        *,
        source_url: str | None = None,
        request_title: str | None = None,
    ) -> dict[str, object]:
        """
        Extract deterministic metadata from a BeautifulSoup document.
        """

        metadata: dict[str, object] = {}

        if source_url:
            metadata["source_url"] = source_url

        title = self._extract_title(
            soup,
            request_title=request_title,
        )

        if title:
            metadata["title"] = title

        description = self._extract_description(soup)

        if description:
            metadata["description"] = description

        author = self._extract_author(soup)

        if author:
            metadata["author"] = author

        language = self._extract_language(soup)

        if language:
            metadata["language"] = language

        canonical_url = self._extract_canonical_url(soup)

        if canonical_url:
            metadata["canonical_url"] = canonical_url

        keywords = self._extract_keywords(soup)

        if keywords:
            metadata["keywords"] = keywords

        open_graph = self._extract_open_graph(soup)

        if open_graph:
            metadata["open_graph"] = open_graph

        article = self._extract_article_metadata(soup)

        if article:
            metadata["article"] = article

        return metadata

    def _extract_title(
        self,
        soup: BeautifulSoup,
        *,
        request_title: str | None,
    ) -> str | None:
        """
        Extract title using deterministic priority:

        <title>
        ↓
        og:title
        ↓
        request title
        """

        title_tag = soup.find("title")

        if isinstance(title_tag, Tag):
            title = self._clean_text(
                title_tag.get_text(
                    " ",
                    strip=True,
                )
            )

            if title:
                return title

        open_graph_title = self._get_meta_content(
            soup,
            property_name="og:title",
        )

        if open_graph_title:
            return open_graph_title

        if request_title:
            return self._clean_text(request_title)

        return None

    def _extract_description(
        self,
        soup: BeautifulSoup,
    ) -> str | None:
        """Extract page description."""

        description = self._get_meta_content(
            soup,
            name="description",
        )

        if description:
            return description

        return self._get_meta_content(
            soup,
            property_name="og:description",
        )

    def _extract_author(
        self,
        soup: BeautifulSoup,
    ) -> str | None:
        """
        Extract author using common metadata conventions.
        """

        author = self._get_meta_content(
            soup,
            name="author",
        )

        if author:
            return author

        return self._get_meta_content(
            soup,
            property_name="article:author",
        )

    def _extract_language(
        self,
        soup: BeautifulSoup,
    ) -> str | None:
        """
        Extract language.

        Priority:
        <html lang="...">
        ↓
        meta content-language
        ↓
        meta language
        """

        html_tag = soup.find("html")

        if isinstance(html_tag, Tag):
            lang = html_tag.get("lang")

            if isinstance(lang, str):
                lang = self._clean_text(lang)

                if lang:
                    return lang

        content_language = self._get_meta_content(
            soup,
            http_equiv="content-language",
        )

        if content_language:
            return content_language

        return self._get_meta_content(
            soup,
            name="language",
        )

    def _extract_canonical_url(
        self,
        soup: BeautifulSoup,
    ) -> str | None:
        """Extract canonical document URL."""

        canonical = soup.find(
            "link",
            rel=lambda value: self._contains_token(
                value,
                "canonical",
            ),
        )

        if not isinstance(canonical, Tag):
            return None

        href = canonical.get("href")

        if not isinstance(href, str):
            return None

        href = href.strip()

        return href or None

    def _extract_keywords(
        self,
        soup: BeautifulSoup,
    ) -> list[str]:
        """Extract and normalize comma-separated keywords."""

        value = self._get_meta_content(
            soup,
            name="keywords",
        )

        if not value:
            return []

        keywords: list[str] = []

        for keyword in value.split(","):
            cleaned = self._clean_text(keyword)

            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)

        return keywords

    def _extract_open_graph(
        self,
        soup: BeautifulSoup,
    ) -> dict[str, str]:
        """Extract supported OpenGraph properties."""

        result: dict[str, str] = {}

        for property_name, output_name in self._OPEN_GRAPH_FIELDS.items():
            value = self._get_meta_content(
                soup,
                property_name=property_name,
            )

            if value:
                result[output_name] = value

        return result

    def _extract_article_metadata(
        self,
        soup: BeautifulSoup,
    ) -> dict[str, object]:
        """Extract article-specific metadata."""

        result: dict[str, object] = {}

        for property_name, output_name in self._ARTICLE_FIELDS.items():
            values = self._get_all_meta_content(
                soup,
                property_name=property_name,
            )

            if not values:
                continue

            if output_name == "author":
                authors = self._unique_clean_values(values)

                if authors:
                    result[output_name] = authors

                continue

            result[output_name] = values[0]

        tags = self._get_all_meta_content(
            soup,
            property_name="article:tag",
        )

        normalized_tags = self._unique_clean_values(tags)

        if normalized_tags:
            result["tags"] = normalized_tags

        return result

    def _get_meta_content(
        self,
        soup: BeautifulSoup,
        *,
        name: str | None = None,
        property_name: str | None = None,
        http_equiv: str | None = None,
    ) -> str | None:
        """Return the first matching meta content value."""

        values = self._get_all_meta_content(
            soup,
            name=name,
            property_name=property_name,
            http_equiv=http_equiv,
        )

        if not values:
            return None

        return values[0]

    def _get_all_meta_content(
        self,
        soup: BeautifulSoup,
        *,
        name: str | None = None,
        property_name: str | None = None,
        http_equiv: str | None = None,
    ) -> list[str]:
        """Return all matching non-empty meta content values."""

        values: list[str] = []

        for tag in soup.find_all("meta"):
            if not isinstance(tag, Tag):
                continue

            if not self._meta_attribute_matches(
                tag,
                attribute="name",
                expected=name,
            ):
                continue

            if not self._meta_attribute_matches(
                tag,
                attribute="property",
                expected=property_name,
            ):
                continue

            if not self._meta_attribute_matches(
                tag,
                attribute="http-equiv",
                expected=http_equiv,
            ):
                continue

            content = tag.get("content")

            if not isinstance(content, str):
                continue

            cleaned = self._clean_text(content)

            if cleaned:
                values.append(cleaned)

        return values

    @staticmethod
    def _meta_attribute_matches(
        tag: Tag,
        *,
        attribute: str,
        expected: str | None,
    ) -> bool:
        """
        Match metadata attributes case-insensitively.

        If expected is None, the attribute is not part of
        the filtering criteria.
        """

        if expected is None:
            return True

        value = tag.get(attribute)

        if not isinstance(value, str):
            return False

        return value.strip().lower() == expected.lower()

    @staticmethod
    def _contains_token(
        value: object,
        token: str,
    ) -> bool:
        """Check whether an HTML attribute contains a token."""

        if isinstance(value, str):
            values: Iterable[str] = value.split()
        elif isinstance(value, list):
            values = (
                item
                for item in value
                if isinstance(item, str)
            )
        else:
            return False

        return any(
            item.lower() == token.lower()
            for item in values
        )

    @classmethod
    def _unique_clean_values(
        cls,
        values: Iterable[str],
    ) -> list[str]:
        """Clean values while preserving deterministic order."""

        result: list[str] = []

        for value in values:
            cleaned = cls._clean_text(value)

            if cleaned and cleaned not in result:
                result.append(cleaned)

        return result

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """Normalize metadata text."""

        text = text.replace(
            "\xa0",
            " ",
        )

        return " ".join(
            text.split()
        ).strip()