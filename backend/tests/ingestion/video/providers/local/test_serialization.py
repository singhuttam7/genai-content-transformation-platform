import base64

import pytest

from app.ingestion.video.providers.local.serialization import (
    Base64FrameSerializer,
    FrameSerializer,
    SerializedFrame,
)
from app.ingestion.video.schemas import VideoFrame


def make_jpeg_frame(
    *,
    frame_index: int = 3,
    timestamp_seconds: float = 4.5,
    image: bytes = b"\xff\xd8\xff\xe0test-image\xff\xd9",
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        image=image,
    )


def make_png_frame(
    *,
    frame_index: int = 5,
    timestamp_seconds: float = 8.25,
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        image=b"\x89PNG\r\n\x1a\nfake-png-data",
    )


class TestSerializedFrame:
    def test_is_empty_returns_false_for_data(self) -> None:
        result = SerializedFrame(
            frame_index=1,
            timestamp_seconds=2.0,
            mime_type="image/jpeg",
            data="YWJj",
        )

        assert result.is_empty is False

    def test_is_empty_returns_true_for_empty_data(self) -> None:
        result = SerializedFrame(
            frame_index=1,
            timestamp_seconds=2.0,
            mime_type="image/jpeg",
            data="",
        )

        assert result.is_empty is True

    def test_serialized_frame_is_immutable(self) -> None:
        result = SerializedFrame(
            frame_index=1,
            timestamp_seconds=2.0,
            mime_type="image/jpeg",
            data="YWJj",
        )

        with pytest.raises(AttributeError):
            result.data = "different"


class TestBase64FrameSerializer:
    def test_serializer_has_stable_name(self) -> None:
        serializer = Base64FrameSerializer()

        assert serializer.name == "base64"

    def test_jpeg_is_serialized(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame()

        result = serializer.serialize(frame)

        assert isinstance(result, SerializedFrame)
        assert result.frame_index == frame.frame_index
        assert result.timestamp_seconds == frame.timestamp_seconds
        assert result.mime_type == "image/jpeg"
        assert result.data == base64.b64encode(frame.image).decode("ascii")
        assert result.is_empty is False

    def test_png_is_serialized(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_png_frame()

        result = serializer.serialize(frame)

        assert result.frame_index == frame.frame_index
        assert result.timestamp_seconds == frame.timestamp_seconds
        assert result.mime_type == "image/png"
        assert result.data == base64.b64encode(frame.image).decode("ascii")

    def test_serialized_data_can_be_decoded(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame()

        result = serializer.serialize(frame)

        decoded = serializer.decode(result)

        assert decoded == frame.image

    def test_decode_round_trip_preserves_png(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_png_frame()

        result = serializer.serialize(frame)

        assert serializer.decode(result) == frame.image

    def test_frame_metadata_is_preserved(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame(
            frame_index=17,
            timestamp_seconds=23.75,
        )

        result = serializer.serialize(frame)

        assert result.frame_index == 17
        assert result.timestamp_seconds == 23.75

    def test_original_frame_is_not_modified(self) -> None:
        serializer = Base64FrameSerializer()

        original_image = b"\xff\xd8original\xff\xd9"

        frame = make_jpeg_frame(
            frame_index=7,
            timestamp_seconds=11.0,
            image=original_image,
        )

        serializer.serialize(frame)

        assert frame.image == original_image
        assert frame.frame_index == 7
        assert frame.timestamp_seconds == 11.0

    def test_none_frame_is_rejected(self) -> None:
        serializer = Base64FrameSerializer()

        with pytest.raises(
            ValueError,
            match="Video frame must be provided",
        ):
            serializer.serialize(None)  # type: ignore[arg-type]

    def test_empty_image_is_rejected(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame(image=b"")

        with pytest.raises(
            ValueError,
            match="image must not be empty",
        ):
            serializer.serialize(frame)

    def test_invalid_image_is_rejected(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame(
            image=b"not-an-image",
        )

        with pytest.raises(
            ValueError,
            match="supported JPEG or PNG",
        ):
            serializer.serialize(frame)

    def test_incomplete_jpeg_is_rejected(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame(
            image=b"\xff\xd8incomplete",
        )

        with pytest.raises(
            ValueError,
            match="supported JPEG or PNG",
        ):
            serializer.serialize(frame)

    def test_invalid_base64_is_rejected(self) -> None:
        serialized = SerializedFrame(
            frame_index=1,
            timestamp_seconds=1.0,
            mime_type="image/jpeg",
            data="%%%invalid%%%",
        )

        with pytest.raises(
            ValueError,
            match="invalid Base64",
        ):
            Base64FrameSerializer.decode(serialized)

    def test_none_serialized_frame_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="Serialized frame must be provided",
        ):
            Base64FrameSerializer.decode(None)  # type: ignore[arg-type]

    def test_empty_serialized_data_is_rejected(self) -> None:
        serialized = SerializedFrame(
            frame_index=1,
            timestamp_seconds=1.0,
            mime_type="image/jpeg",
            data="",
        )

        with pytest.raises(
            ValueError,
            match="data must not be empty",
        ):
            Base64FrameSerializer.decode(serialized)

    def test_protocol_can_be_implemented(self) -> None:
        serializer: FrameSerializer = Base64FrameSerializer()

        result = serializer.serialize(
            make_jpeg_frame(),
        )

        assert isinstance(result, SerializedFrame)

    def test_multiple_frames_remain_independent(self) -> None:
        serializer = Base64FrameSerializer()

        first = make_jpeg_frame(
            frame_index=1,
            timestamp_seconds=2.0,
            image=b"\xff\xd8first\xff\xd9",
        )

        second = make_jpeg_frame(
            frame_index=2,
            timestamp_seconds=4.0,
            image=b"\xff\xd8second\xff\xd9",
        )

        first_result = serializer.serialize(first)
        second_result = serializer.serialize(second)

        assert first_result.frame_index == 1
        assert second_result.frame_index == 2

        assert serializer.decode(first_result) == first.image
        assert serializer.decode(second_result) == second.image

    def test_base64_output_is_ascii(self) -> None:
        serializer = Base64FrameSerializer()

        result = serializer.serialize(
            make_jpeg_frame(),
        )

        result.data.encode("ascii")

    def test_serialization_is_deterministic(self) -> None:
        serializer = Base64FrameSerializer()

        frame = make_jpeg_frame()

        first = serializer.serialize(frame)
        second = serializer.serialize(frame)

        assert first == second

    def test_serialized_frame_contains_no_runtime_specific_fields(self) -> None:
        serializer = Base64FrameSerializer()

        result = serializer.serialize(
            make_jpeg_frame(),
        )

        assert set(result.__dataclass_fields__) == {
            "frame_index",
            "timestamp_seconds",
            "mime_type",
            "data",
        }