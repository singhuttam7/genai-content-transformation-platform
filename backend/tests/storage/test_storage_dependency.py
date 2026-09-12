from app.storage.dependencies import get_storage_service
from app.storage.service import StorageService


def test_storage_dependency():
    service = get_storage_service()

    print(
        "Service Type:",
        type(service).__name__,
    )

    print(
        "Provider Type:",
        type(service.provider).__name__,
    )

    print(
        "Service Test:",
        "OK"
        if isinstance(service, StorageService)
        else "FAILED",
    )

    print(
        "Provider Test:",
        "OK"
        if service.provider is not None
        else "FAILED",
    )


if __name__ == "__main__":
    test_storage_dependency()