import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StorageConfig:
    """Configure the S3-compatible object store."""

    endpoint_url: str = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    access_key: str = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY", "rustfsadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("S3_SECRET_KEY", "rustfsadmin"))
    region: str = field(default_factory=lambda: os.getenv("S3_REGION", "us-east-1"))
    bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "falcon-images"))
    presign_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("S3_PRESIGN_TTL", "3600")))
    public_endpoint_url: str | None = field(default_factory=lambda: os.getenv("S3_PUBLIC_ENDPOINT_URL") or None)

    @property
    def public_url(self) -> str:
        """Endpoint used for signing links handed to clients."""
        return self.public_endpoint_url or self.endpoint_url
