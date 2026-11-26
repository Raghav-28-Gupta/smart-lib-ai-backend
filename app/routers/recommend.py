from fastapi import APIRouter, Depends

from app.core.security import require_ai_token
from app.ml.recommend.engine import recommend_batch
from app.schemas.recommend import BatchRequest, BatchResponse

# Every route here needs the shared secret (404 while AI_TOKEN is unset, 401 when the header is wrong).
router = APIRouter(prefix="/recommend", dependencies=[Depends(require_ai_token)])


@router.post("/batch", response_model=BatchResponse)
def recommend_for_all(payload: BatchRequest) -> BatchResponse:
    """Top picks for every student in the payload, fitted from the payload alone (nothing is stored).

    A plain `def`, so FastAPI runs this handler, the CPU-bound fit, in its thread pool. The request body is
    read, parsed and validated on the event loop before that, so a cap-sized body holds the loop for a while.
    """
    # Measured once in the final review: `/predict/no-show` waited at most about half a second behind a
    # cap-sized batch, in process on a 16-core laptop. A small host was not measured and would wait longer.
    model_version, users = recommend_batch(payload.books, payload.interactions, payload.limit)
    return BatchResponse(model_version=model_version, users=users)
