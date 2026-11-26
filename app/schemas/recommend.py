from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

RecommendationMethod = Literal["content", "collaborative", "hybrid"]

# The caps of POST /recommend/batch (spec 2026-11-25-book-recommender-design, section 1). The Node
# backend sends at most as much and truncates to the most recent interactions, so a 422 here means a bug.
MAX_BOOKS = 5_000
MAX_INTERACTIONS = 200_000
MAX_PAIR_WEIGHT = 3.0
MAX_LIMIT = 50
DEFAULT_LIMIT = 10


class BookIn(BaseModel):
    """One catalog book. The text fields may be empty: a book with no text simply never matches."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    author: str
    genre: str | None = None
    description: str | None = None


class InteractionIn(BaseModel):
    """One student's interaction with one book (a borrow, a renewal, a reservation).

    `weight` is how strong the signal is: the backend sends 1 per loan, +0.5 when renewed, 0.5 per
    reservation, and a pair that repeats is summed here and capped at MAX_PAIR_WEIGHT.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    book_id: str
    weight: float = Field(gt=0.0, le=MAX_PAIR_WEIGHT, allow_inf_nan=False)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    books: list[BookIn] = Field(min_length=1, max_length=MAX_BOOKS)
    interactions: list[InteractionIn] = Field(max_length=MAX_INTERACTIONS)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)

    @model_validator(mode="after")
    def book_ids_are_unique(self) -> Self:
        # The message names no id: a 422 never echoes a value of the input back.
        if len({book.id for book in self.books}) != len(self.books):
            raise ValueError("book ids must be unique")
        return self


class RecommendedBook(BaseModel):
    book_id: str
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    method: RecommendationMethod


class UserRecommendations(BaseModel):
    user_id: str
    results: list[RecommendedBook]


class BatchResponse(BaseModel):
    model_version: str
    users: list[UserRecommendations]
