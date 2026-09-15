class FalconError(Exception):
    """Base class for interdomain errors."""

    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(FalconError):
    """Input failed ingestion validation."""

    code = "validation_error"


class NotFoundError(FalconError):
    """The requested object does not exist."""

    code = "not_found"


class StorageError(FalconError):
    """Object storage or the task bus is unavailable."""

    code = "storage_error"
