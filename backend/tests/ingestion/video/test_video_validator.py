import pytest

from app.ingestion.video.schemas import VideoProcessingStatus
from app.ingestion.video.validator import VideoSecurityValidator


def mp4_bytes() -> bytes:
    """
    Minimal byte sequence containing an MP4-style ftyp box.
    This is sufficient for the pre-processing signature check.
    """
    return (
        b"\x00\x00\x00\x18"
        b"ftyp"
        b"isom"
        b"\x00\x00\x02\x00"
        b"isomiso2mp41"
    )


def mov_bytes() -> bytes:
    return (
        b"\x00\x00\x00\x18"
        b"ftyp"
        b"qt  "
        b"\x00\x00\x00\x00"
        b"qt  "
        b"qt  "
    )


def webm_bytes() -> bytes:
    return b"\x1A\x45\xDF\xA3" + b"\x00" * 32


def avi_bytes() -> bytes:
    return b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 32


def test_valid_mp4_is_accepted():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="sample.mp4",
        mime_type="video/mp4",
    )

    assert result.valid is True
    assert result.status == VideoProcessingStatus.COMPLETED
    assert result.extension == ".mp4"
    assert result.mime_type == "video/mp4"


def test_valid_mov_is_accepted():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mov_bytes(),
        filename="sample.mov",
        mime_type="video/quicktime",
    )

    assert result.valid is True
    assert result.extension == ".mov"


def test_valid_webm_is_accepted():
    validator = VideoSecurityValidator()

    result = validator.validate(
        webm_bytes(),
        filename="sample.webm",
        mime_type="video/webm",
    )

    assert result.valid is True
    assert result.extension == ".webm"


def test_valid_avi_is_accepted():
    validator = VideoSecurityValidator()

    result = validator.validate(
        avi_bytes(),
        filename="sample.avi",
        mime_type="video/x-msvideo",
    )

    assert result.valid is True
    assert result.extension == ".avi"


def test_empty_video_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        b"",
        filename="sample.mp4",
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert result.status == VideoProcessingStatus.FAILED
    assert "empty" in result.reason.lower()


def test_video_larger_than_limit_is_rejected():
    validator = VideoSecurityValidator(
        max_size_bytes=10,
    )

    result = validator.validate(
        b"x" * 11,
        filename="sample.mp4",
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert "maximum allowed size" in result.reason


def test_missing_filename_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert "filename" in result.reason.lower()


@pytest.mark.parametrize(
    "filename",
    [
        "sample.txt",
        "sample.pdf",
        "sample.exe",
        "sample",
    ],
)
def test_unsupported_extension_is_rejected(filename):
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename=filename,
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert "extension" in result.reason.lower()


@pytest.mark.parametrize(
    "filename",
    [
        "../sample.mp4",
        "..\\sample.mp4",
        "folder/sample.mp4",
        "folder\\sample.mp4",
    ],
)
def test_path_traversal_filename_is_rejected(filename):
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename=filename,
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert "path" in result.reason.lower()


def test_null_byte_filename_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="sample\x00.mp4",
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert "null byte" in result.reason.lower()


def test_mime_type_mismatch_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="sample.mp4",
        mime_type="video/webm",
    )

    assert result.valid is False
    assert "compatible" in result.reason.lower()


def test_non_video_mime_type_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="sample.mp4",
        mime_type="text/plain",
    )

    assert result.valid is False
    assert "mime" in result.reason.lower()


def test_invalid_mp4_signature_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        b"this is not an mp4 file",
        filename="sample.mp4",
        mime_type="video/mp4",
    )

    assert result.valid is False
    assert "signature" in result.reason.lower()


def test_invalid_webm_signature_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        b"this is not webm",
        filename="sample.webm",
        mime_type="video/webm",
    )

    assert result.valid is False
    assert "signature" in result.reason.lower()


def test_invalid_avi_signature_is_rejected():
    validator = VideoSecurityValidator()

    result = validator.validate(
        b"this is not avi",
        filename="sample.avi",
        mime_type="video/x-msvideo",
    )

    assert result.valid is False
    assert "signature" in result.reason.lower()


def test_mime_parameter_is_optional():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="sample.mp4",
    )

    assert result.valid is True
    assert result.mime_type is None


def test_mime_parameters_are_normalized():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="sample.mp4",
        mime_type="video/mp4; charset=binary",
    )

    assert result.valid is True
    assert result.mime_type == "video/mp4"


def test_extension_matching_is_case_insensitive():
    validator = VideoSecurityValidator()

    result = validator.validate(
        mp4_bytes(),
        filename="SAMPLE.MP4",
        mime_type="video/mp4",
    )

    assert result.valid is True
    assert result.extension == ".mp4"


def test_size_is_reported():
    validator = VideoSecurityValidator()

    media = mp4_bytes()

    result = validator.validate(
        media,
        filename="sample.mp4",
        mime_type="video/mp4",
    )

    assert result.size_bytes == len(media)


def test_validator_rejects_invalid_max_size_configuration():
    with pytest.raises(ValueError):
        VideoSecurityValidator(max_size_bytes=0)