import io
from dataclasses import dataclass

import orjson
from PIL import Image, UnidentifiedImageError

from common.src.configs.constants import (
    ALLOWED_IMAGE_TYPES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MIN_SIDE_PX,
)
from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError

# Global Pillow state
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Metadata read off an uploaded image."""

    width: int
    height: int
    image_format: str
    content_type: str
    size_bytes: int


def validate_image(data: bytes, declared_content_type: str | None = None) -> ImageInfo:
    """Check an uploaded image at the ingestion stage.

    Args:
        data: raw bytes of the upload.
        declared_content_type: Content-Type sent with the request, if any.

    Returns:
        The image's metadata.

    Raises:
        ValidationError: the file is empty, oversized, corrupt, or an unsupported format.
    """
    if not data:
        msg = "empty file"
        raise ValidationError(msg)

    if len(data) > MAX_IMAGE_BYTES:
        msg = f"file exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB"
        raise ValidationError(msg, details={"size_bytes": len(data)})

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError) as exc:
        msg = "file is not a valid image"
        raise ValidationError(msg) from exc

    content_type = f"image/{'jpeg' if image_format == 'JPEG' else image_format.lower()}"
    if content_type not in ALLOWED_IMAGE_TYPES:
        msg = f"unsupported image format: {image_format or 'unknown'}"
        raise ValidationError(msg, details={"allowed": sorted(ALLOWED_IMAGE_TYPES)})

    if declared_content_type and declared_content_type not in ALLOWED_IMAGE_TYPES:
        msg = f"unsupported content type: {declared_content_type}"
        raise ValidationError(msg, details={"allowed": sorted(ALLOWED_IMAGE_TYPES)})

    if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
        msg = f"image is smaller than {MIN_SIDE_PX}px per side"
        raise ValidationError(msg, details={"size": [width, height]})

    return ImageInfo(
        width=width,
        height=height,
        image_format=image_format,
        content_type=content_type,
        size_bytes=len(data),
    )


def validate_bbox(bbox: BBox, info: ImageInfo) -> BBox:
    """Check the box against the real image and trim it to the frame.

    Args:
        bbox: box as supplied by the caller.
        info: metadata from validate_image function.

    Returns:
        A box guaranteed to lie inside the image.

    Raises:
        ValidationError: the box lies outside the frame or degenerates once trimmed.
    """
    if bbox.x >= info.width or bbox.y >= info.height:
        msg = "bbox lies outside the image"
        details_dict = {"image": [info.width, info.height], "bbox": bbox.model_dump()}
        raise ValidationError(msg, details=details_dict)
    try:
        return bbox.clamp(info.width, info.height)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def parse_bbox(raw: str) -> BBox:
    """Parse the bbox form field. Accepts either an object or a ``[x, y, width, height]`` array.

    Args:
        raw: the raw form value.

    Returns:
        The parsed box.

    Raises:
        ValidationError: the value is not a usable bbox.
    """
    try:
        if raw.lstrip().startswith("["):
            x, y, width, height = orjson.loads(raw)
            return BBox(x=x, y=y, width=width, height=height)
        return BBox.model_validate_json(raw)
    except ValidationError:
        raise
    except Exception as exc:
        msg = f"could not parse bbox: {exc}"
        raise ValidationError(msg, details={"raw": raw[:200]}) from exc
