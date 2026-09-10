from app.ingestion.detectors.default import DefaultInputDetector
from app.ingestion.schemas import IngestionRequest, InputType


detector = DefaultInputDetector()


def test_text() -> None:
    request = IngestionRequest(
        input_type=InputType.TEXT,
        content="Artificial intelligence is transforming communication.",
    )

    assert detector.detect(request) == InputType.TEXT
    print("TEXT detection: OK")


def test_prompt() -> None:
    request = IngestionRequest(
        input_type=InputType.PROMPT,
        content="Create an executive summary.",
    )

    assert detector.detect(request) == InputType.PROMPT
    print("PROMPT detection: OK")


def test_pdf() -> None:
    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="report.pdf",
        mime_type="application/pdf",
        content=b"fake pdf",
    )

    assert detector.detect(request) == InputType.PDF
    print("PDF detection: OK")


def test_docx() -> None:
    request = IngestionRequest(
        input_type=InputType.DOCX,
        filename="report.docx",
        mime_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        content=b"fake docx",
    )

    assert detector.detect(request) == InputType.DOCX
    print("DOCX detection: OK")


def test_image() -> None:
    request = IngestionRequest(
        input_type=InputType.IMAGE,
        filename="image.png",
        mime_type="image/png",
        content=b"fake image",
    )

    assert detector.detect(request) == InputType.IMAGE
    print("IMAGE detection: OK")


def test_audio() -> None:
    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="audio.mp3",
        mime_type="audio/mpeg",
        content=b"fake audio",
    )

    assert detector.detect(request) == InputType.AUDIO
    print("AUDIO detection: OK")


def test_video() -> None:
    request = IngestionRequest(
        input_type=InputType.VIDEO,
        filename="video.mp4",
        mime_type="video/mp4",
        content=b"fake video",
    )

    assert detector.detect(request) == InputType.VIDEO
    print("VIDEO detection: OK")


def test_url() -> None:
    request = IngestionRequest(
        input_type=InputType.URL,
        url="https://example.com/article",
    )

    assert detector.detect(request) == InputType.URL
    print("URL detection: OK")


def expect_value_error(request: IngestionRequest) -> None:
    try:
        detector.detect(request)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError was not raised."
    )


def test_invalid_url() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.URL,
            url="not-a-url",
        )
    )

    print("Invalid URL rejection: OK")


def test_ftp_url() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.URL,
            url="ftp://example.com/file",
        )
    )

    print("FTP URL rejection: OK")


def test_mime_mismatch() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.PDF,
            filename="image.png",
            mime_type="image/png",
            content=b"fake",
        )
    )

    print("MIME mismatch rejection: OK")


def test_extension_mismatch() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.PDF,
            filename="image.png",
            content=b"fake",
        )
    )

    print("Extension mismatch rejection: OK")


def test_mime_extension_mismatch() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.PDF,
            filename="report.pdf",
            mime_type="image/png",
            content=b"fake",
        )
    )

    print("MIME/extension mismatch rejection: OK")


def test_uppercase_extension() -> None:
    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="REPORT.PDF",
        mime_type="application/pdf",
        content=b"fake",
    )

    assert detector.detect(request) == InputType.PDF
    print("Uppercase extension normalization: OK")


def test_mime_parameters() -> None:
    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="report.pdf",
        mime_type="application/pdf; charset=binary",
        content=b"fake",
    )

    assert detector.detect(request) == InputType.PDF
    print("MIME parameter normalization: OK")


def test_filename_path_rejection() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.PDF,
            filename="../../report.pdf",
            content=b"fake",
        )
    )

    print("Filename path rejection: OK")


def test_unsupported_file_metadata() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.PDF,
            filename="report.unknown",
            mime_type="application/x-unknown",
            content=b"fake",
        )
    )

    print("Unsupported file metadata rejection: OK")


def test_text_txt_compatibility() -> None:
    request = IngestionRequest(
        input_type=InputType.TEXT,
        filename="notes.txt",
        mime_type="text/plain",
        content="Some text",
    )

    assert detector.detect(request) == InputType.TEXT
    print("TEXT/TXT compatibility: OK")


def test_missing_file_metadata() -> None:
    expect_value_error(
        IngestionRequest(
            input_type=InputType.PDF,
            content=b"fake pdf",
        )
    )

    print("Missing file metadata rejection: OK")


def main() -> None:
    print()
    print("==============================================")
    print("INPUT DETECTOR HARDENING TESTS")
    print("==============================================")
    print()

    test_text()
    test_prompt()
    test_pdf()
    test_docx()
    test_image()
    test_audio()
    test_video()
    test_url()

    test_invalid_url()
    test_ftp_url()
    test_mime_mismatch()
    test_extension_mismatch()
    test_mime_extension_mismatch()

    test_uppercase_extension()
    test_mime_parameters()
    test_filename_path_rejection()
    test_unsupported_file_metadata()
    test_text_txt_compatibility()
    test_missing_file_metadata()

    print()
    print("==============================================")
    print("INPUT DETECTOR HARDENING: ALL TESTS PASSED")
    print("==============================================")


if __name__ == "__main__":
    main()