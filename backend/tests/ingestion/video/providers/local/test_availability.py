import pytest

from app.ingestion.video.providers.local.availability import (
    RuntimeAvailability,
    RuntimeAvailabilityChecker,
    RuntimeAvailabilityStatus,
)
from app.ingestion.video.providers.local.config import LocalVisionConfig


class FakeRuntimeAvailabilityChecker:
    name = "fake-runtime"

    def __init__(
        self,
        result: RuntimeAvailability,
    ) -> None:
        self.result = result
        self.received_config: LocalVisionConfig | None = None

    async def check(
        self,
        config: LocalVisionConfig,
    ) -> RuntimeAvailability:
        self.received_config = config
        return self.result


def make_config() -> LocalVisionConfig:
    return LocalVisionConfig(
        model="test-vision-model",
        base_url="http://localhost:9999",
        timeout_seconds=30.0,
        max_retries=2,
        temperature=0.2,
        keep_alive="5m",
    )


class TestRuntimeAvailabilityStatus:
    def test_status_values(self) -> None:
        assert RuntimeAvailabilityStatus.AVAILABLE.value == "available"
        assert (
            RuntimeAvailabilityStatus.RUNTIME_UNAVAILABLE.value
            == "runtime_unavailable"
        )
        assert (
            RuntimeAvailabilityStatus.MODEL_UNAVAILABLE.value
            == "model_unavailable"
        )
        assert RuntimeAvailabilityStatus.UNKNOWN.value == "unknown"


class TestRuntimeAvailability:
    def test_available_when_runtime_and_model_are_available(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.AVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=True,
        )

        assert result.available is True

    def test_not_available_when_runtime_is_unavailable(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.RUNTIME_UNAVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=False,
            model_available=False,
            errors=("Runtime is unreachable.",),
        )

        assert result.available is False

    def test_not_available_when_model_is_unavailable(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.MODEL_UNAVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=False,
            errors=("Requested model is unavailable.",),
        )

        assert result.available is False

    def test_unknown_is_not_available(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.UNKNOWN,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=False,
            model_available=False,
            errors=("Availability could not be determined.",),
        )

        assert result.available is False

    def test_errors_default_to_empty_tuple(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.AVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=True,
        )

        assert result.errors == ()

    def test_errors_are_immutable(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.RUNTIME_UNAVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=False,
            model_available=False,
            errors=("Runtime unavailable.",),
        )

        assert isinstance(result.errors, tuple)

    def test_configuration_values_are_preserved(self) -> None:
        config = make_config()

        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.AVAILABLE,
            runtime_name="fake-runtime",
            model=config.model,
            runtime_available=True,
            model_available=True,
        )

        assert result.model == config.model


class TestRuntimeAvailabilityChecker:
    @pytest.mark.asyncio
    async def test_protocol_can_be_implemented(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.AVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=True,
        )

        checker = FakeRuntimeAvailabilityChecker(result)

        config = make_config()

        availability = await checker.check(config)

        assert availability is result
        assert checker.received_config is config
        assert checker.name == "fake-runtime"

    @pytest.mark.asyncio
    async def test_checker_receives_complete_configuration(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.AVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=True,
        )

        checker = FakeRuntimeAvailabilityChecker(result)

        config = make_config()

        await checker.check(config)

        assert checker.received_config is not None
        assert checker.received_config.model == "test-vision-model"
        assert checker.received_config.base_url == "http://localhost:9999"
        assert checker.received_config.timeout_seconds == 30.0
        assert checker.received_config.max_retries == 2
        assert checker.received_config.temperature == 0.2
        assert checker.received_config.keep_alive == "5m"


class TestRuntimeAvailabilityContract:
    def test_runtime_unavailable_requires_runtime_false(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.RUNTIME_UNAVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=False,
            model_available=False,
        )

        assert result.runtime_available is False
        assert result.available is False

    def test_model_unavailable_can_have_runtime_available(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.MODEL_UNAVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=False,
        )

        assert result.runtime_available is True
        assert result.model_available is False
        assert result.available is False

    def test_available_requires_both_capabilities(self) -> None:
        result = RuntimeAvailability(
            status=RuntimeAvailabilityStatus.AVAILABLE,
            runtime_name="fake-runtime",
            model="test-vision-model",
            runtime_available=True,
            model_available=True,
        )

        assert result.runtime_available is True
        assert result.model_available is True
        assert result.available is True

    def test_checker_is_runtime_agnostic(self) -> None:
        checker = FakeRuntimeAvailabilityChecker(
            RuntimeAvailability(
                status=RuntimeAvailabilityStatus.AVAILABLE,
                runtime_name="fake-runtime",
                model="test-vision-model",
                runtime_available=True,
                model_available=True,
            )
        )

        assert isinstance(checker.name, str)
        assert checker.name == "fake-runtime"