import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QdrantConfig:
    """Configure the Qdrant connection."""

    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
