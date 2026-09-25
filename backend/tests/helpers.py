import io
import uuid

import numpy as np
from PIL import Image

from common.src.configs.schemas import BBox, Candidate, SearchResult, TaskStatus

BOX = BBox(x=10, y=10, width=64, height=48)
BOX_JSON = BOX.model_dump_json()


def make_image(seed: int = 1, size: tuple[int, int] = (128, 96), image_format: str = "PNG") -> bytes:
    """Encode a noise image; different seeds give visually unrelated images."""
    pixels = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=image_format)
    return buffer.getvalue()


def make_candidates(count: int) -> list[Candidate]:
    """Candidates at ranks 1..count with descending scores, shaped like inference's."""
    return [
        Candidate(
            image_id=f"img-{rank}",
            score=1.0 - rank / 100,
            rank=rank,
            image_path=f"gallery/img-{rank}.png",
            image_url=f"http://s3.test/gallery/img-{rank}.png?X-Amz-Signature=expires-in-an-hour",
            vehicle_id=f"veh-{rank}",
            camera_id="cam-1",
        )
        for rank in range(1, count + 1)
    ]


def make_result(
    task_id: str | None = None,
    *,
    count: int = 3,
    rejected: bool = False,
    status: TaskStatus = TaskStatus.DONE,
    error: str | None = None,
) -> SearchResult:
    """A finished result as inference would publish it."""
    candidates = [] if rejected or status is TaskStatus.FAILED else make_candidates(count)
    return SearchResult(
        task_id=task_id or uuid.uuid4().hex,
        status=status,
        candidates=candidates,
        top_score=None if status is TaskStatus.FAILED else 0.42 if rejected else 0.99,
        rejected=rejected,
        model_name="stub",
        latency_ms=12.5,
        error=error,
    )
