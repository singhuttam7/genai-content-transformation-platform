from __future__ import annotations

from app.models_ai.embeddings.adapters.sentence_transformer import (
    SentenceTransformerAdapter,
)
from app.models_ai.embeddings.schemas import EmbeddingModelConfig


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_config(
    *,
    batch_size: int = 32,
    normalized: bool = False,
    options: dict[str, object] | None = None,
) -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name=MODEL_NAME,
        batch_size=batch_size,
        normalized=normalized,
        options={} if options is None else options,
    )


def create_adapter(
    *,
    batch_size: int = 32,
    normalized: bool = False,
    options: dict[str, object] | None = None,
) -> SentenceTransformerAdapter:
    return SentenceTransformerAdapter(
        create_config(
            batch_size=batch_size,
            normalized=normalized,
            options=options,
        ),
        model_factory=lambda _: object(),
    )


def test_default_constructor_options_are_empty() -> None:
    adapter = create_adapter()

    assert adapter._constructor_options() == {}


def test_device_is_constructor_option() -> None:
    adapter = create_adapter(
        options={
            "device": "cpu",
        },
    )

    assert adapter._constructor_options() == {
        "device": "cpu",
    }


def test_encode_options_are_separated_from_constructor_options() -> None:
    adapter = create_adapter(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    assert adapter._constructor_options() == {
        "device": "cpu",
    }

    encode_options = adapter._encode_options()

    assert encode_options["show_progress_bar"] is False
    assert "device" not in encode_options


def test_batch_size_is_always_taken_from_configuration() -> None:
    adapter = create_adapter(
        batch_size=16,
        options={
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    encode_options = adapter._encode_options()

    assert encode_options["batch_size"] == 16


def test_normalization_is_derived_from_configuration() -> None:
    adapter = create_adapter(
        normalized=True,
    )

    encode_options = adapter._encode_options()

    assert encode_options["normalize_embeddings"] is True


def test_explicit_normalization_option_overrides_generated_default() -> None:
    adapter = create_adapter(
        normalized=True,
        options={
            "encode": {
                "normalize_embeddings": False,
            },
        },
    )

    encode_options = adapter._encode_options()

    assert encode_options["normalize_embeddings"] is False


def test_explicit_batch_size_option_overrides_generated_default() -> None:
    adapter = create_adapter(
        batch_size=32,
        options={
            "encode": {
                "batch_size": 8,
            },
        },
    )

    encode_options = adapter._encode_options()

    assert encode_options["batch_size"] == 8


def test_unknown_top_level_options_are_not_passed_to_encode() -> None:
    adapter = create_adapter(
        options={
            "device": "cpu",
            "constructor_option": "value",
        },
    )

    encode_options = adapter._encode_options()

    assert "device" not in encode_options
    assert "constructor_option" not in encode_options


def test_unknown_nested_encode_options_are_preserved() -> None:
    adapter = create_adapter(
        options={
            "encode": {
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        },
    )

    encode_options = adapter._encode_options()

    assert encode_options["show_progress_bar"] is False
    assert encode_options["convert_to_numpy"] is True


def test_constructor_options_do_not_mutate_configuration() -> None:
    options = {
        "device": "cpu",
    }

    adapter = create_adapter(options=options)

    constructor_options = adapter._constructor_options()
    constructor_options["device"] = "cuda"

    assert options["device"] == "cpu"
    assert adapter.model_config.options["device"] == "cpu"


def test_encode_options_do_not_mutate_nested_configuration() -> None:
    options = {
        "encode": {
            "show_progress_bar": False,
        },
    }

    adapter = create_adapter(options=options)

    encode_options = adapter._encode_options()
    encode_options["show_progress_bar"] = True

    assert (
        adapter.model_config.options["encode"]["show_progress_bar"]
        is False
    )


def test_constructor_and_encode_options_are_independent() -> None:
    adapter = create_adapter(
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    constructor_options = adapter._constructor_options()
    encode_options = adapter._encode_options()

    constructor_options["device"] = "cuda"
    encode_options["show_progress_bar"] = True

    assert adapter.model_config.options["device"] == "cpu"
    assert (
        adapter.model_config.options["encode"]["show_progress_bar"]
        is False
    )


def test_empty_encode_options_remain_empty_when_no_defaults_are_needed() -> None:
    adapter = create_adapter()

    encode_options = adapter._encode_options()

    assert encode_options["batch_size"] == 32
    assert "normalize_embeddings" not in encode_options