from app.storage.factory import create_storage_provider
from app.storage.files.local import LocalStorageProvider


def test_storage_factory():
    provider = create_storage_provider()

    print(
        "Provider Type:",
        type(provider).__name__,
    )

    print(
        "Factory Test:",
        "OK"
        if isinstance(provider, LocalStorageProvider)
        else "FAILED",
    )


if __name__ == "__main__":
    test_storage_factory()