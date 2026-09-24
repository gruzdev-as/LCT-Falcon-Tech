import os
from dataclasses import dataclass, field
from urllib.parse import quote


@dataclass(frozen=True)
class PostgresConfig:
    """Configure the Postgres connection."""

    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "falcon"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "falcon"))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "falcon"))
    pool_size: int = field(default_factory=lambda: int(os.getenv("POSTGRES_POOL_SIZE", "10")))
    echo: bool = field(default_factory=lambda: os.getenv("POSTGRES_ECHO", "").lower() == "true")

    @property
    def dsn(self) -> str:
        """SQLAlchemy URL for the async driver."""
        override = os.getenv("POSTGRES_DSN")
        if override:
            return override
        credentials = f"{quote(self.user, safe='')}:{quote(self.password, safe='')}"
        return f"postgresql+asyncpg://{credentials}@{self.host}:{self.port}/{self.database}"
