from bs4 import BeautifulSoup

from app.ingestion.parsers.html_metadata import HTMLMetadataExtractor


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(
        html,
        "html.parser",
    )


def test_extract_title() -> None:
    html = """
    <html>
        <head>
            <title>Critical Security Advisory</title>
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["title"] == "Critical Security Advisory"


def test_extract_description() -> None:
    html = """
    <html>
        <head>
            <meta
                name="description"
                content="Critical vulnerability discovered."
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["description"] == (
        "Critical vulnerability discovered."
    )


def test_extract_author() -> None:
    html = """
    <html>
        <head>
            <meta
                name="author"
                content="Security Research Team"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["author"] == (
        "Security Research Team"
    )


def test_extract_language_from_html() -> None:
    html = """
    <html lang="en">
        <head></head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["language"] == "en"


def test_extract_canonical_url() -> None:
    html = """
    <html>
        <head>
            <link
                rel="canonical"
                href="https://example.com/advisory"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["canonical_url"] == (
        "https://example.com/advisory"
    )


def test_extract_keywords() -> None:
    html = """
    <html>
        <head>
            <meta
                name="keywords"
                content="security, vulnerability, security, threat intelligence"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["keywords"] == [
        "security",
        "vulnerability",
        "threat intelligence",
    ]


def test_extract_open_graph_metadata() -> None:
    html = """
    <html>
        <head>
            <meta
                property="og:title"
                content="Cybersecurity Alert"
            >
            <meta
                property="og:description"
                content="A critical vulnerability was discovered."
            >
            <meta
                property="og:type"
                content="article"
            >
            <meta
                property="og:site_name"
                content="Security Portal"
            >
            <meta
                property="og:image"
                content="https://example.com/image.png"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["open_graph"] == {
        "title": "Cybersecurity Alert",
        "description": (
            "A critical vulnerability was discovered."
        ),
        "type": "article",
        "site_name": "Security Portal",
        "image": "https://example.com/image.png",
    }


def test_extract_article_metadata() -> None:
    html = """
    <html>
        <head>
            <meta
                property="article:section"
                content="Cybersecurity"
            >
            <meta
                property="article:published_time"
                content="2026-09-11T10:30:00Z"
            >
            <meta
                property="article:modified_time"
                content="2026-09-11T12:00:00Z"
            >
            <meta
                property="article:author"
                content="Research Team"
            >
            <meta
                property="article:author"
                content="Research Team"
            >
            <meta
                property="article:tag"
                content="Threat Intelligence"
            >
            <meta
                property="article:tag"
                content="Cybersecurity"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["article"] == {
        "section": "Cybersecurity",
        "published_time": "2026-09-11T10:30:00Z",
        "modified_time": "2026-09-11T12:00:00Z",
        "author": ["Research Team"],
        "tags": [
            "Threat Intelligence",
            "Cybersecurity",
        ],
    }


def test_title_falls_back_to_open_graph() -> None:
    html = """
    <html>
        <head>
            <meta
                property="og:title"
                content="OpenGraph Security Alert"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["title"] == (
        "OpenGraph Security Alert"
    )


def test_title_falls_back_to_request_title() -> None:
    html = """
    <html>
        <head></head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
        request_title="Requested Document Title",
    )

    assert metadata["title"] == (
        "Requested Document Title"
    )


def test_source_url_is_preserved() -> None:
    html = """
    <html>
        <head></head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
        source_url="https://example.com/report",
    )

    assert metadata["source_url"] == (
        "https://example.com/report"
    )


def test_metadata_attributes_are_case_insensitive() -> None:
    html = """
    <html>
        <head>
            <meta
                NAME="DESCRIPTION"
                content="Security report"
            >
            <meta
                PROPERTY="OG:SITE_NAME"
                content="Security Portal"
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert metadata["description"] == "Security report"

    assert metadata["open_graph"]["site_name"] == (
        "Security Portal"
    )


def test_empty_metadata_is_ignored() -> None:
    html = """
    <html>
        <head>
            <title>   </title>
            <meta
                name="description"
                content="   "
            >
            <meta
                name="keywords"
                content=", , "
            >
        </head>
    </html>
    """

    extractor = HTMLMetadataExtractor()

    metadata = extractor.extract(
        make_soup(html),
    )

    assert "title" not in metadata
    assert "description" not in metadata
    assert "keywords" not in metadata