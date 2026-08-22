from typing import Literal

from pydantic import BaseModel, Field

RecommendationMethod = Literal["content", "collaborative", "hybrid"]


class RecommendRequest(BaseModel):
    user_id: str
    limit: int = Field(default=10, ge=1, le=50)


class RecommendedBook(BaseModel):
    book_id: str
    score: float
    method: RecommendationMethod


class RecommendResponse(BaseModel):
    user_id: str
    results: list[RecommendedBook]
    model_version: str
