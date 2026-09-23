import io

from PIL import Image, UnidentifiedImageError

from common.src.configs.constants import MAX_IMAGE_PIXELS
from common.src.configs.schemas import BBox
from common.src.exceptions import ValidationError

# Same decompression-bomb ceiling the backend validated against
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def crop_vehicle(data: bytes, bbox: BBox) -> Image.Image:
    """Decode an image and cut out the vehicle.

    No EXIF transpose: the backend validated the bbox against the raw decoded frame,
    so the coordinates refer to that frame.

    Args:
        data: encoded image bytes as stored at ingestion.
        bbox: vehicle box in absolute pixels.

    Returns:
        The cropped region, in the image's original mode.

    Raises:
        ValidationError: the bytes do not decode, or the box degenerates once trimmed
            to the image.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            box = bbox.clamp(*image.size)
            return image.crop(box.to_xyxy())
    except (UnidentifiedImageError, OSError) as exc:
        msg = "stored image cannot be decoded"
        raise ValidationError(msg) from exc
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
