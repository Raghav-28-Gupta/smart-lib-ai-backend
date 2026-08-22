from fastapi import APIRouter

from app.schemas.recommend import RecommendRequest, RecommendResponse

router = APIRouter(prefix="/recommend")


@router.post("/books", response_model=RecommendResponse)
def recommend_books(payload: RecommendRequest) -> RecommendResponse:
    # Placeholder until the hybrid recommender lands in Phase 3.
    return RecommendResponse(user_id=payload.user_id, results=[], model_version="stub")
