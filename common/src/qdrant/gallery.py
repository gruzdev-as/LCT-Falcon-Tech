import uuid
from typing import Final

from qdrant_client.models import PointStruct

from common.src.configs.schemas import BBox

# Point ids must be uint64 or UUID, so image ids go through uuid5. Never change it:
# a new namespace remaps every id and the gallery is indexed a second time.
POINT_NAMESPACE: Final[uuid.UUID] = uuid.UUID("4e3c7a67-36f1-4148-9b2e-e775a7990da8")

MODEL_VERSION_FIELD: Final[str] = "model_version"


def point_id(image_id: str) -> str:
    """Map a dataset image id onto a Qdrant point id."""
    return str(uuid.uuid5(POINT_NAMESPACE, image_id))


def gallery_point(
    *,
    image_id: str,
    image_path: str,
    bbox: BBox,
    vehicle_id: str | None,
    camera_id: str | None,
    model_version: str,
    vector: list[float],
) -> PointStruct:
    """Assemble one gallery point with the payload the search result reads.

    ``image_id`` is the real id, since the point id is derived from it; ``model_version``
    says which weights produced the vector.
    """
    return PointStruct(
        id=point_id(image_id),
        vector=vector,
        payload={
            "image_id": image_id,
            "image_path": image_path,
            "bbox": bbox.model_dump(),
            "vehicle_id": vehicle_id,
            "camera_id": camera_id,
            MODEL_VERSION_FIELD: model_version,
        },
    )
