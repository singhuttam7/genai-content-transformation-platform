from __future__ import annotations

import pytest

from app.ingestion.parsers.html import HTMLDocumentProcessor
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


def make_request(
    content: bytes,
) -> IngestionRequest:
    return IngestionRequest(
        input_type=InputType.URL,
        url="https://example.com/article",
        title=None,
        mime_type="text/html",
        content=content,
    )


@pytest.mark.asyncio
async def test_html_processor_extracts_basic_structure() -> None:
    html = b"""
    <html>
        <body>
            <h1>Cybersecurity Report</h1>
            <p>This is the first paragraph.</p>
            <h2>Impact</h2>
            <p>The incident affected several systems.</p>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert result.text == (
        "Cybersecurity Report\n\n"
        "This is the first paragraph.\n\n"
        "Impact\n\n"
        "The incident affected several systems."
    )

    assert len(result.blocks) == 4

    assert result.blocks[0].block_type == (
        ContentBlockType.HEADING
    )

    assert result.blocks[1].block_type == (
        ContentBlockType.PARAGRAPH
    )

    assert result.blocks[2].block_type == (
        ContentBlockType.HEADING
    )

    assert result.blocks[3].block_type == (
        ContentBlockType.PARAGRAPH
    )


@pytest.mark.asyncio
async def test_html_processor_extracts_list() -> None:
    html = b"""
    <html>
        <body>
            <h1>Systems</h1>
            <ul>
                <li>System A</li>
                <li>System B</li>
            </ul>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 2

    assert result.blocks[1].block_type == (
        ContentBlockType.LIST
    )

    assert result.blocks[1].content == (
        "System A\nSystem B"
    )


@pytest.mark.asyncio
async def test_html_processor_extracts_quote() -> None:
    html = b"""
    <html>
        <body>
            <blockquote>
                This is an important statement.
            </blockquote>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].block_type == (
        ContentBlockType.QUOTE
    )


@pytest.mark.asyncio
async def test_html_processor_extracts_code() -> None:
    html = b"""
    <html>
        <body>
            <pre>print("hello")</pre>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].block_type == (
        ContentBlockType.CODE
    )


@pytest.mark.asyncio
async def test_html_processor_removes_non_content_elements() -> None:
    html = b"""
    <html>
        <head>
            <style>
                body { display: none; }
            </style>
        </head>
        <body>
            <script>
                alert("malicious");
            </script>

            <p>Visible content.</p>

            <iframe>
                hidden content
            </iframe>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert result.text == "Visible content."
    assert "malicious" not in result.text
    assert "hidden content" not in result.text


@pytest.mark.asyncio
async def test_html_processor_preserves_block_order() -> None:
    html = b"""
    <html>
        <body>
            <h1>Title</h1>
            <p>Paragraph one.</p>
            <h2>Section</h2>
            <p>Paragraph two.</p>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert [
        block.order
        for block in result.blocks
    ] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_html_processor_rejects_empty_content() -> None:
    processor = HTMLDocumentProcessor()

    with pytest.raises(
        ValueError,
        match="HTML content cannot be empty",
    ):
        await processor.process(
            make_request(b"")
        )


@pytest.mark.asyncio
async def test_html_processor_rejects_content_without_meaningful_blocks() -> None:
    html = b"""
    <html>
        <body>
            <script>console.log("test");</script>
            <style>body {}</style>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    with pytest.raises(
        ValueError,
        match="No meaningful content",
    ):
        await processor.process(
            make_request(html)
        )


@pytest.mark.asyncio
async def test_html_processor_rejects_wrong_input_type() -> None:
    request = IngestionRequest(
        input_type=InputType.TEXT,
        content="hello",
    )

    processor = HTMLDocumentProcessor()

    with pytest.raises(
        ValueError,
        match="only supports URL input",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_nested_inline_elements_are_not_duplicated() -> None:
    html = b"""
    <html>
        <body>
            <p>
                This is <strong>important</strong>
                information about
                <a href="/report">the incident</a>.
            </p>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].content == (
        "This is important information about the incident."
    )


@pytest.mark.asyncio
async def test_nested_list_is_extracted_as_single_block() -> None:
    html = b"""
    <html>
        <body>
            <ul>
                <li>System A</li>
                <li>
                    System B
                    <ul>
                        <li>Database</li>
                        <li>API</li>
                    </ul>
                </li>
            </ul>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 1
    assert result.blocks[0].block_type == (
        ContentBlockType.LIST
    )

    assert result.blocks[0].content == (
        "System A\nSystem B Database API"
    )


@pytest.mark.asyncio
async def test_table_is_extracted_as_atomic_block() -> None:
    html = b"""
    <html>
        <body>
            <table>
                <tr>
                    <th>System</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>API</td>
                    <td>Compromised</td>
                </tr>
            </table>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 1

    assert result.blocks[0].block_type == (
        ContentBlockType.TABLE
    )

    assert result.blocks[0].content == (
        "System | Status\n"
        "API | Compromised"
    )

    assert result.blocks[0].metadata["rows"] == 2


@pytest.mark.asyncio
async def test_figure_preserves_image_metadata() -> None:
    html = b"""
    <html>
        <body>
            <figure>
                <img
                    src="/images/network.png"
                    alt="Network attack diagram"
                >
                <figcaption>
                    Attack path
                </figcaption>
            </figure>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert len(result.blocks) == 1

    block = result.blocks[0]

    assert block.block_type == (
        ContentBlockType.IMAGE
    )

    assert block.content == (
        "Network attack diagram\nAttack path"
    )

    assert block.metadata["alt"] == (
        "Network attack diagram"
    )

    assert block.metadata["caption"] == (
        "Attack path"
    )

    assert block.metadata["src"] == (
        "/images/network.png"
    )


@pytest.mark.asyncio
async def test_html_entities_are_decoded() -> None:
    html = b"""
    <html>
        <body>
            <p>
                Security &amp; Privacy &lt;Report&gt;
            </p>
        </body>
    </html>
    """

    processor = HTMLDocumentProcessor()

    result = await processor.process(
        make_request(html)
    )

    assert result.blocks[0].content == (
        "Security & Privacy <Report>"
    )