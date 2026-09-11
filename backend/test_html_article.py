from bs4 import BeautifulSoup
import pytest

from app.ingestion.parsers.html_article import (
    HTMLArticleExtractor,
)


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(
        html,
        "html.parser",
    )


def test_article_tag_is_selected() -> None:
    html = """
    <html>
        <body>
            <nav>
                Home About Contact
            </nav>

            <article>
                <h1>Security Incident Report</h1>

                <p>
                    A significant security incident was discovered
                    during routine infrastructure monitoring.
                </p>

                <p>
                    Security engineers investigated the affected
                    systems and identified the source of the issue.
                </p>
            </article>

            <footer>
                Copyright 2026
            </footer>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert result.extraction_method == "semantic_article"
    assert "Security Incident Report" in result.text
    assert "security incident" in result.text.lower()


def test_main_tag_is_selected() -> None:
    html = """
    <html>
        <body>
            <header>
                Website Header
            </header>

            <main>
                <h1>Threat Intelligence Report</h1>

                <p>
                    Analysts discovered suspicious activity across
                    several systems during the investigation.
                </p>

                <p>
                    Additional analysis confirmed the presence of
                    malicious network activity.
                </p>
            </main>

            <footer>
                Footer information
            </footer>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert result.extraction_method == "semantic_main"
    assert "Threat Intelligence Report" in result.text


def test_best_candidate_is_selected() -> None:
    html = """
    <html>
        <body>
            <div class="sidebar">
                <p>
                    Short sidebar information about this website.
                </p>
            </div>

            <div class="article-content">
                <h1>Cybersecurity Advisory</h1>

                <p>
                    A critical vulnerability has been identified
                    in a widely deployed software component.
                </p>

                <p>
                    Organizations should immediately review their
                    systems and apply the recommended mitigations.
                </p>

                <p>
                    Additional monitoring is recommended until the
                    affected systems have been secured.
                </p>
            </div>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert result.extraction_method == (
        "scored_structural_candidate"
    )

    assert "Cybersecurity Advisory" in result.text
    assert result.metadata["paragraph_count"] == 3


def test_navigation_is_penalized() -> None:
    html = """
    <html>
        <body>
            <div class="navigation">
                <p>
                    Home Products Services Security Reports
                    Documentation Contact About
                </p>
            </div>

            <article>
                <h1>Security Advisory</h1>

                <p>
                    A vulnerability was discovered in the
                    authentication component of the platform.
                </p>

                <p>
                    Administrators should apply the recommended
                    security update as soon as possible.
                </p>
            </article>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert "Security Advisory" in result.text
    assert result.extraction_method == "semantic_article"


def test_sidebar_is_penalized() -> None:
    html = """
    <html>
        <body>
            <div class="sidebar">
                <p>
                    Related stories and recommended resources
                    are listed in this section.
                </p>
            </div>

            <main>
                <h1>Incident Analysis</h1>

                <p>
                    Investigators analyzed the incident and
                    reconstructed the complete attack sequence.
                </p>

                <p>
                    The analysis revealed multiple compromised
                    systems within the affected environment.
                </p>
            </main>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert "Incident Analysis" in result.text
    assert result.extraction_method == "semantic_main"


def test_advertisement_is_penalized() -> None:
    html = """
    <html>
        <body>
            <div class="advertisement">
                <p>
                    Buy our premium security monitoring service
                    today and protect your organization.
                </p>
            </div>

            <article>
                <h1>Threat Report</h1>

                <p>
                    Researchers identified a new threat targeting
                    enterprise infrastructure.
                </p>

                <p>
                    The campaign uses multiple techniques to gain
                    access and maintain persistence.
                </p>
            </article>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert "Threat Report" in result.text
    assert result.extraction_method == "semantic_article"


def test_link_density_is_penalized() -> None:
    html = """
    <html>
        <body>
            <div class="link-section">
                <p>
                    <a href="/one">Security</a>
                    <a href="/two">Threats</a>
                    <a href="/three">Reports</a>
                    <a href="/four">Advisories</a>
                    <a href="/five">Research</a>
                    <a href="/six">Analysis</a>
                </p>
            </div>

            <div class="content">
                <h1>Security Research</h1>

                <p>
                    Researchers conducted an extensive analysis
                    of the security environment and discovered
                    several important vulnerabilities.
                </p>

                <p>
                    The findings indicate that organizations
                    should strengthen their security controls.
                </p>
            </div>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert "Security Research" in result.text
    assert result.metadata["link_density"] < 1.0


def test_short_candidates_are_rejected() -> None:
    html = """
    <html>
        <body>
            <article>
                Too short.
            </article>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    with pytest.raises(ValueError):
        extractor.extract(
            make_soup(html),
        )


def test_fallback_to_structural_content() -> None:
    html = """
    <html>
        <body>
            <div>
                <p>
                    Security researchers discovered a significant
                    vulnerability affecting several enterprise
                    systems and services.
                </p>

                <p>
                    The investigation confirmed that the issue
                    could allow unauthorized access to affected
                    infrastructure.
                </p>
            </div>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert result.extraction_method in {
        "scored_structural_candidate",
        "body_fallback",
    }

    assert "Security researchers" in result.text


def test_no_content_raises_error() -> None:
    html = """
    <html>
        <body>
            <nav>
                Home
            </nav>

            <footer>
                Footer
            </footer>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    with pytest.raises(ValueError):
        extractor.extract(
            make_soup(html),
        )


def test_extraction_diagnostics_are_recorded() -> None:
    html = """
    <html>
        <body>
            <article>
                <h1>Security Investigation</h1>

                <p>
                    Investigators examined suspicious activity
                    detected within the enterprise network.
                </p>

                <p>
                    The investigation identified compromised
                    infrastructure and several attack indicators.
                </p>
            </article>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert result.candidate_count >= 1
    assert result.selected_score > 0

    assert "paragraph_count" in result.metadata
    assert "heading_count" in result.metadata
    assert "link_density" in result.metadata
    assert "semantic_bonus" in result.metadata
    assert "boilerplate_penalty" in result.metadata


def test_article_class_receives_positive_signal() -> None:
    html = """
    <html>
        <body>
            <div class="article-body">
                <h1>Cybersecurity Research</h1>

                <p>
                    Researchers analyzed a sophisticated attack
                    against critical infrastructure systems.
                </p>

                <p>
                    The investigation revealed several techniques
                    used by the attackers during the campaign.
                </p>
            </div>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert result.extraction_method == (
        "scored_structural_candidate"
    )

    assert result.metadata["semantic_bonus"] == 0.0


def test_result_text_is_normalized() -> None:
    html = """
    <html>
        <body>
            <article>
                <h1>
                    Security &amp; Privacy
                </h1>

                <p>
                    This is an important security report
                    about the incident .
                </p>

                <p>
                    Additional findings were confirmed .
                </p>
            </article>
        </body>
    </html>
    """

    extractor = HTMLArticleExtractor()

    result = extractor.extract(
        make_soup(html),
    )

    assert "Security & Privacy" in result.text
    assert "incident ." not in result.text
    assert "incident." in result.text