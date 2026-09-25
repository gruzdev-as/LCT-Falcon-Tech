from typing import Annotated

from fastapi import APIRouter, Form, Response, status

from backend.app.dependencies import DbSession
from backend.src.configs.responses import INVALID_INPUT, STILL_PROCESSING, UNKNOWN_TASK
from backend.src.configs.schemas import SearchAccepted, SearchForm, SearchPending
from backend.src.services import search as search_service
from common.src.configs.schemas import SearchResult
from common.src.exceptions import NotFoundError

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", status_code=status.HTTP_202_ACCEPTED, responses=INVALID_INPUT)
async def submit_search(
    form: Annotated[SearchForm, Form(media_type="multipart/form-data")],
    session: DbSession,
) -> SearchAccepted:
    """Submit an image for re-identification.

    Validates the image and bounding box, stores the original, and queues a ReID task.
    Returns immediately with a task id, poll `GET /search/{task_id}` for the top-k nearest.
    \f
    Raises:
        ValidationError: the image or the bbox failed validation.
        StorageError: the original could not be stored.
    """
    task_id = await search_service.submit(
        session,
        data=await form.file.read(),
        filename=form.file.filename or "query.jpg",
        bbox_raw=form.bbox,
        top_k=form.top_k,
        declared_content_type=form.file.content_type,
    )
    await session.commit()
    return SearchAccepted(task_id=task_id)


@router.get("/{task_id}", responses=STILL_PROCESSING | UNKNOWN_TASK)
async def get_search_result(task_id: str, response: Response, session: DbSession) -> SearchResult | SearchPending:
    """Retrieve a search result.

    Returns the ranked candidates once inference has published them.
    While the task is still in flight the response is `202` with a processing status;
    An unknown or expired task id gives `404`.
    \f
    Raises:
        NotFoundError: neither a result nor a live task exists for this id.
    """
    result = await search_service.fetch_result(session, task_id)
    await session.commit()

    if result is not None:
        return result

    if await search_service.task_exists(task_id):
        response.status_code = status.HTTP_202_ACCEPTED
        return SearchPending(task_id=task_id)

    msg = f"unknown or expired task: {task_id}"
    raise NotFoundError(msg)
