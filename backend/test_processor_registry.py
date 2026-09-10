from app.ingestion.registry import create_processor_router
from app.ingestion.schemas import InputType


def main() -> None:
    router = create_processor_router()

    assert router.supports(InputType.TEXT)
    assert router.supports(InputType.PROMPT)
    assert router.supports(InputType.TXT)
    assert router.supports(InputType.MARKDOWN)

    assert not router.supports(InputType.PDF)
    assert not router.supports(InputType.DOCX)

    text_processor = router.get_processor(InputType.TEXT)
    prompt_processor = router.get_processor(InputType.PROMPT)
    txt_processor = router.get_processor(InputType.TXT)
    markdown_processor = router.get_processor(
        InputType.MARKDOWN
    )

    assert text_processor is prompt_processor
    assert txt_processor is markdown_processor

    print("TEXT processor registration: OK")
    print("PROMPT processor registration: OK")
    print("TXT processor registration: OK")
    print("MARKDOWN processor registration: OK")
    print("Unsupported PDF registration: OK")
    print("Unsupported DOCX registration: OK")
    print()
    print("Processor registry: ALL TESTS PASSED")


if __name__ == "__main__":
    main()