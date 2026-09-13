import pytest

from app.ingestion.video.providers.local.config import LocalVisionConfig


class TestLocalVisionConfig:
    def test_default_configuration(self) -> None:
        config = LocalVisionConfig()

        assert config.model == "gemma3:4b"
        assert config.base_url == "http://localhost:11434"
        assert config.timeout_seconds == 120.0
        assert config.max_retries == 1
        assert config.temperature == 0.1
        assert config.keep_alive == "5m"

    def test_custom_configuration(self) -> None:
        config = LocalVisionConfig(
            model="custom-vision-model",
            base_url="http://127.0.0.1:11434/",
            timeout_seconds=60.0,
            max_retries=3,
            temperature=0.5,
            keep_alive="10m",
        )

        assert config.model == "custom-vision-model"
        assert config.base_url == "http://127.0.0.1:11434"
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 3
        assert config.temperature == 0.5
        assert config.keep_alive == "10m"

    def test_model_is_trimmed(self) -> None:
        config = LocalVisionConfig(
            model="  gemma3:4b  ",
        )

        assert config.model == "gemma3:4b"

    def test_base_url_is_trimmed_and_trailing_slash_removed(self) -> None:
        config = LocalVisionConfig(
            base_url="  http://localhost:11434///  ",
        )

        assert config.base_url == "http://localhost:11434"

    def test_keep_alive_is_trimmed(self) -> None:
        config = LocalVisionConfig(
            keep_alive="  10m  ",
        )

        assert config.keep_alive == "10m"

    def test_empty_model_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="Vision model must not be empty",
        ):
            LocalVisionConfig(model="   ")

    def test_empty_base_url_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="Vision runtime base URL must not be empty",
        ):
            LocalVisionConfig(base_url="   ")

    def test_zero_timeout_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="timeout must be greater than zero",
        ):
            LocalVisionConfig(timeout_seconds=0)

    def test_negative_timeout_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="timeout must be greater than zero",
        ):
            LocalVisionConfig(timeout_seconds=-1)

    def test_negative_retries_are_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="max retries must be zero or greater",
        ):
            LocalVisionConfig(max_retries=-1)

    def test_temperature_below_range_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="temperature must be between 0.0 and 2.0",
        ):
            LocalVisionConfig(temperature=-0.1)

    def test_temperature_above_range_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="temperature must be between 0.0 and 2.0",
        ):
            LocalVisionConfig(temperature=2.1)

    def test_temperature_boundary_values_are_allowed(self) -> None:
        assert LocalVisionConfig(temperature=0.0).temperature == 0.0
        assert LocalVisionConfig(temperature=2.0).temperature == 2.0

    def test_none_keep_alive_is_allowed(self) -> None:
        config = LocalVisionConfig(keep_alive=None)

        assert config.keep_alive is None

    def test_empty_keep_alive_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="keep-alive must not be empty",
        ):
            LocalVisionConfig(keep_alive="   ")

    def test_configuration_is_immutable(self) -> None:
        config = LocalVisionConfig()

        with pytest.raises(AttributeError):
            config.model = "another-model"