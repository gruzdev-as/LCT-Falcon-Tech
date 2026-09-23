import logging
import uuid
from functools import lru_cache
from typing import Any

import anyio.to_thread
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from common.src.configs.constants import QUERY_PREFIX
from common.src.exceptions import NotFoundError, StorageError
from common.src.storage.config import StorageConfig

logger = logging.getLogger(__name__)

_BOTO_CONFIG = Config(signature_version="s3v4", s3={"addressing_style": "path"})
_MISSING_KEY_CODES = frozenset({"NoSuchKey", "404"})


@lru_cache
def get_config() -> StorageConfig:
    """Return the cached storage configuration."""
    return StorageConfig()


@lru_cache
def _client(endpoint_url: str) -> Any:
    """Build a boto3 client for one endpoint."""
    config = get_config()
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        region_name=config.region,
        config=_BOTO_CONFIG,
    )


def make_path(filename: str, prefix: str = QUERY_PREFIX, name: str | None = None) -> str:
    """Build an object key for a new image, preserving its extension."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    return f"{prefix}/{name or uuid.uuid4().hex}.{suffix}"


async def ensure_bucket() -> None:
    """Create the bucket if it does not exist yet."""
    await anyio.to_thread.run_sync(_ensure_bucket_sync)


def _ensure_bucket_sync() -> None:
    config = get_config()
    client = _client(config.endpoint_url)
    try:
        client.head_bucket(Bucket=config.bucket)
    except ClientError:
        client.create_bucket(Bucket=config.bucket)
        logger.info("Created bucket %s", config.bucket)


async def save(key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    """Store an image. boto3 is blocking, so the call is pushed to a worker thread."""
    config = get_config()
    client = _client(config.endpoint_url)
    try:
        await anyio.to_thread.run_sync(
            lambda: client.put_object(Bucket=config.bucket, Key=key, Body=data, ContentType=content_type)
        )
    except (ClientError, BotoCoreError) as exc:
        msg = f"failed to store image {key}"
        raise StorageError(msg) from exc
    return key


async def load(key: str) -> bytes:
    """Read an image back.

    Raises:
        NotFoundError: there is no object under this key.
        StorageError: the store itself is unreachable or failed.
    """
    config = get_config()
    client = _client(config.endpoint_url)
    try:
        response = await anyio.to_thread.run_sync(lambda: client.get_object(Bucket=config.bucket, Key=key))
        return await anyio.to_thread.run_sync(response["Body"].read)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
            msg = f"image {key} does not exist"
            raise NotFoundError(msg) from exc
        msg = f"failed to read image {key}"
        raise StorageError(msg) from exc
    except BotoCoreError as exc:
        msg = f"failed to read image {key}"
        raise StorageError(msg) from exc


def url_for(key: str, ttl: int | None = None) -> str:
    """Build a temporary link the frontend can load the image from."""
    config = get_config()
    client = _client(config.public_url)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.bucket, "Key": key},
        ExpiresIn=ttl or config.presign_ttl_seconds,
    )
