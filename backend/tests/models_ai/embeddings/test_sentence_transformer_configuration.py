from __future__ import annotations

from app.models_ai.embeddings.adapters.sentence_transformer import (
    SentenceTransformerAdapter,
)
from app.models_ai.embeddings.schemas import EmbeddingModelConfig


def create_config(
    *,
    provider: str = "sentence-transformers",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    expected_dimension: int | None = 384,
    normalized: bool = False,
    batch_size: int = 32,
    options: dict | None = None,
) -> EmbeddingModelConfig:
    """Create a standard embedding model configuration."""

    return EmbeddingModelConfig(
        provider=provider,
        model_name=model_name,
        expected_dimension=expected_dimension,
        normalized=normalized,
        batch_size=batch_size,
        options={} if options is None else options,
    )


def test_default_configuration_values() -> None:
    config = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert config.provider == "sentence-transformers"
    assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.expected_dimension is None
    assert config.normalized is False
    assert config.batch_size == 32
    assert config.options == {}


def test_default_configuration_is_compatible_with_adapter() -> None:
    config = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    assert adapter.model_config == config


def test_adapter_preserves_provider_configuration() -> None:
    config = create_config(
        provider="custom-sentence-transformer",
        model_name="custom-model",
        expected_dimension=768,
        normalized=True,
        batch_size=16,
        options={
            "device": "cpu",
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    assert adapter.model_config.provider == "custom-sentence-transformer"
    assert adapter.model_config.model_name == "custom-model"
    assert adapter.model_config.expected_dimension == 768
    assert adapter.model_config.normalized is True
    assert adapter.model_config.batch_size == 16
    assert adapter.model_config.options == {
        "device": "cpu",
    }


def test_provider_cannot_be_blank() -> None:
    try:
        EmbeddingModelConfig(
            provider="   ",
            model_name="test-model",
        )
    except ValueError:
        return

    raise AssertionError("Expected blank provider to be rejected.")


def test_model_name_cannot_be_blank() -> None:
    try:
        EmbeddingModelConfig(
            provider="sentence-transformers",
            model_name="   ",
        )
    except ValueError:
        return

    raise AssertionError("Expected blank model name to be rejected.")


def test_expected_dimension_must_be_positive() -> None:
    try:
        EmbeddingModelConfig(
            provider="sentence-transformers",
            model_name="test-model",
            expected_dimension=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected non-positive dimension to be rejected."
    )


def test_batch_size_must_be_positive() -> None:
    try:
        EmbeddingModelConfig(
            provider="sentence-transformers",
            model_name="test-model",
            batch_size=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected non-positive batch size to be rejected."
    )


def test_configuration_is_frozen() -> None:
    config = create_config()

    try:
        config.batch_size = 16
    except Exception:
        return

    raise AssertionError(
        "Expected EmbeddingModelConfig to be immutable."
    )


def test_options_are_preserved() -> None:
    options = {
        "device": "cpu",
        "show_progress_bar": False,
    }

    config = create_config(
        options=options,
    )

    assert config.options == options


def test_constructor_options_are_separated_from_encode_options() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "show_progress_bar": False,
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    assert constructor_options == {
        "device": "cpu",
    }

    assert "device" not in encode_options
    assert encode_options["show_progress_bar"] is False


def test_nested_encode_options_are_supported() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["show_progress_bar"] is False
    assert encode_options["convert_to_numpy"] is True
    assert "device" not in encode_options


def test_flat_encode_options_are_supported_for_backward_compatibility() -> None:
    config = create_config(
        options={
            "show_progress_bar": False,
            "convert_to_numpy": True,
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["show_progress_bar"] is False
    assert encode_options["convert_to_numpy"] is True


def test_normalized_configuration_adds_normalization_option() -> None:
    config = create_config(
        normalized=True,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["normalize_embeddings"] is True


def test_non_normalized_configuration_does_not_force_normalization() -> None:
    config = create_config(
        normalized=False,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert "normalize_embeddings" not in encode_options


def test_explicit_normalization_option_has_precedence() -> None:
    config = create_config(
        normalized=True,
        options={
            "normalize_embeddings": False,
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["normalize_embeddings"] is False


def test_nested_normalization_option_has_precedence_over_flat_option() -> None:
    config = create_config(
        normalized=True,
        options={
            "normalize_embeddings": False,
            "encode": {
                "normalize_embeddings": True,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["normalize_embeddings"] is True


def test_batch_size_configuration_is_passed_to_encode() -> None:
    config = create_config(
        batch_size=16,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["batch_size"] == 16


def test_explicit_batch_size_option_has_precedence() -> None:
    config = create_config(
        batch_size=32,
        options={
            "batch_size": 8,
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["batch_size"] == 8


def test_nested_batch_size_option_has_precedence_over_flat_option() -> None:
    config = create_config(
        batch_size=32,
        options={
            "batch_size": 16,
            "encode": {
                "batch_size": 8,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["batch_size"] == 8


def test_custom_options_are_preserved() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "show_progress_bar": False,
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    assert constructor_options["device"] == "cpu"
    assert encode_options["show_progress_bar"] is False
    assert "device" not in encode_options


def test_nested_custom_options_are_preserved() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert encode_options["show_progress_bar"] is False
    assert encode_options["convert_to_numpy"] is True
    assert "device" not in encode_options


def test_constructor_options_do_not_mutate_configuration() -> None:
    options = {
        "device": "cpu",
    }

    config = create_config(
        options=options,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    constructor_options = adapter._constructor_options()

    constructor_options["device"] = "cuda"

    assert config.options["device"] == "cpu"
    assert options["device"] == "cpu"


def test_encode_options_do_not_mutate_configuration() -> None:
    options = {
        "device": "cpu",
        "encode": {
            "show_progress_bar": False,
        },
    }

    config = create_config(
        options=options,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    encode_options["show_progress_bar"] = True
    encode_options["batch_size"] = 8

    assert config.options["encode"]["show_progress_bar"] is False
    assert "batch_size" not in config.options["encode"]


def test_encode_options_are_independent_between_calls() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    first = adapter._encode_options()
    second = adapter._encode_options()

    first["show_progress_bar"] = True

    assert second["show_progress_bar"] is False
    assert config.options["encode"]["show_progress_bar"] is False


def test_constructor_options_are_independent_between_calls() -> None:
    config = create_config(
        options={
            "device": "cpu",
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    first = adapter._constructor_options()
    second = adapter._constructor_options()

    first["device"] = "cuda"

    assert second["device"] == "cpu"
    assert config.options["device"] == "cpu"


def test_nested_options_are_independent_between_calls() -> None:
    config = create_config(
        options={
            "encode": {
                "show_progress_bar": False,
                "custom": {
                    "value": 1,
                },
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    first = adapter._encode_options()
    second = adapter._encode_options()

    first["custom"]["value"] = 999

    assert second["custom"]["value"] == 1
    assert config.options["encode"]["custom"]["value"] == 1


def test_constructor_and_encode_options_are_independent() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    constructor_options["device"] = "cuda"
    encode_options["show_progress_bar"] = True

    assert constructor_options["device"] == "cuda"
    assert encode_options["show_progress_bar"] is True

    assert config.options["device"] == "cpu"
    assert config.options["encode"]["show_progress_bar"] is False


def test_model_options_are_passed_without_mutating_configuration() -> None:
    options = {
        "device": "cpu",
        "show_progress_bar": False,
    }

    config = create_config(
        options=options,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    assert constructor_options == {
        "device": "cpu",
    }

    assert encode_options == {
        "show_progress_bar": False,
        "batch_size": 32,
    }

    assert config.options == options


def test_nested_model_options_are_passed_without_mutating_configuration() -> None:
    options = {
        "device": "cpu",
        "encode": {
            "show_progress_bar": False,
            "convert_to_numpy": True,
        },
    }

    config = create_config(
        options=options,
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    assert constructor_options == {
        "device": "cpu",
    }

    assert encode_options == {
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "batch_size": 32,
    }

    assert config.options == options


def test_unknown_top_level_options_are_not_passed_to_encode() -> None:
    config = create_config(
        options={
            "device": "cpu",
            "unknown_option": "value",
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert "device" not in encode_options
    assert "unknown_option" not in encode_options


def test_unknown_nested_encode_options_are_preserved() -> None:
    config = create_config(
        options={
            "encode": {
                "some_future_sentence_transformer_option": "value",
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    encode_options = adapter._encode_options()

    assert (
        encode_options[
            "some_future_sentence_transformer_option"
        ]
        == "value"
    )


def test_nested_encode_options_must_be_a_dictionary() -> None:
    config = create_config(
        options={
            "encode": "invalid",
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=lambda _: object(),
    )

    try:
        adapter._encode_options()
    except Exception as exc:
        assert isinstance(
            exc,
            Exception,
        )
        return

    raise AssertionError(
        "Expected invalid nested encode options to be rejected."
    )


def test_building_options_does_not_load_model() -> None:
    calls: list[str] = []

    def factory(_: str) -> object:
        calls.append("loaded")
        return object()

    config = create_config(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        config,
        model_factory=factory,
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    assert constructor_options == {
        "device": "cpu",
    }

    assert encode_options["show_progress_bar"] is False
    assert calls == []
    assert adapter.is_loaded is False