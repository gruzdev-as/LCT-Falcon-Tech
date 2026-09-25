import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from common.src.configs.schemas import BBox, Candidate, SearchResult, TaskStatus
from common.src.db.models import SearchCandidate, SearchQuery

logger = logging.getLogger(__name__)


async def insert_query(
    session: AsyncSession,
    *,
    task_id: str,
    image_path: str,
    bbox: BBox,
    top_k: int,
    image_width: int,
    image_height: int,
    image_format: str,
    content_type: str,
    size_bytes: int,
) -> None:
    """Record an accepted search request.

    Args:
        session: open session; not committed here.
        task_id: the id the client will poll.
        image_path: object key of the stored original.
        bbox: the box after clamping to the real frame.
        top_k: how many candidates were asked for.
        image_width: decoded width, px.
        image_height: decoded height, px.
        image_format: Pillow's format name, uppercase.
        content_type: MIME type derived from the real format.
        size_bytes: size of the upload.
    """
    statement = insert(SearchQuery).values(
        task_id=task_id,
        image_path=image_path,
        bbox_x=bbox.x,
        bbox_y=bbox.y,
        bbox_width=bbox.width,
        bbox_height=bbox.height,
        top_k=top_k,
        image_width=image_width,
        image_height=image_height,
        image_format=image_format,
        content_type=content_type,
        size_bytes=size_bytes,
        status=TaskStatus.PENDING,
    )
    # The id is a fresh uuid4, so a conflict means a retry of the same request
    await session.execute(statement.on_conflict_do_nothing(index_elements=[SearchQuery.task_id]))


async def finalize_search(session: AsyncSession, result: SearchResult) -> bool:
    """Record the outcome of a finished search, exactly once.

    Args:
        session: open session; not committed here.
        result: the result inference published.

    Returns:
        True when this call wrote the outcome,
        False when the row was already finalized or was never inserted at submit time.
    """
    statement = (
        update(SearchQuery)
        .where(SearchQuery.task_id == result.task_id, SearchQuery.finalized_at.is_(None))
        .values(
            status=result.status,
            top_score=result.top_score,
            rejected=result.rejected,
            candidate_count=len(result.candidates),
            model_name=result.model_name,
            latency_ms=result.latency_ms,
            error=result.error,
            finished_at=result.finished_at,
            finalized_at=datetime.now(UTC),
        )
        # Nothing of this entity is loaded in the session, so skip the sync pass.
        .execution_options(synchronize_session=False)
    )
    updated = await session.execute(statement)
    if updated.rowcount != 1:
        logger.debug("Search %s was already finalized, or was never recorded", result.task_id)
        return False

    if result.candidates:
        await _insert_candidates(session, result.task_id, result.candidates)
    return True


async def _insert_candidates(session: AsyncSession, task_id: str, candidates: Sequence[Candidate]) -> None:
    """Write the ranked candidates of one search."""
    rows = [
        {
            "task_id": task_id,
            "rank": candidate.rank,
            "image_id": candidate.image_id,
            "score": candidate.score,
            "image_path": candidate.image_path,
            "vehicle_id": candidate.vehicle_id,
            "camera_id": candidate.camera_id,
        }
        for candidate in candidates
    ]
    statement = insert(SearchCandidate).values(rows)
    await session.execute(
        statement.on_conflict_do_nothing(index_elements=[SearchCandidate.task_id, SearchCandidate.rank])
    )
