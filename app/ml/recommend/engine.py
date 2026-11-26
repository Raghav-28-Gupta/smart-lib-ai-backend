"""The batch recommender: top picks per student from one payload (spec 2026-11-25-book-recommender-design).

Stateless and deterministic: every call fits from the books and interactions it is given, with fixed
seeds, so the same payload gives the same answer. Two parts are blended per student: a content part
(TF-IDF, `content.py`) and a collaborative part (implicit ALS, `collaborative.py`) that only runs when
the payload holds enough history.
"""

from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from app.ml.recommend.collaborative import fit_als
from app.ml.recommend.content import content_scores, fit_book_vectors
from app.schemas.recommend import (
    MAX_PAIR_WEIGHT,
    BookIn,
    InteractionIn,
    RecommendationMethod,
    RecommendedBook,
    UserRecommendations,
)

MODEL_VERSION_CONTENT = "tfidf-1"
MODEL_VERSION_HYBRID = "tfidf-als-1"  # the collaborative part ran

# A student's collaborative weight grows by this much per distinct book they have, up to the maximum:
# a student with little history gets mostly content picks, one with a long history leans on the others.
COLLABORATIVE_WEIGHT_PER_BOOK = 0.1
MAX_COLLABORATIVE_WEIGHT = 0.7

# Students scored at a time. A dense score block is chunk x books floats (512 x 5,000 x 8 bytes = 20 MB
# at the largest catalog the contract allows) and a few live at once (content, collaborative, blended),
# so this memory stays bounded however many students there are.
USER_CHUNK = 512


def recommend_batch(
    books: Sequence[BookIn],
    interactions: Sequence[InteractionIn],
    limit: int,
) -> tuple[str, list[UserRecommendations]]:
    """Return `(model_version, users)`: up to `limit` picks for each student, users in id order.

    A student appears only with at least one pick. An interaction on a book that is not in `books`
    is ignored, so a student with no known book has no profile and is absent.
    """
    book_ids = [book.id for book in books]
    user_ids, weights = _weight_matrix({book_id: i for i, book_id in enumerate(book_ids)}, interactions)
    if not user_ids:
        return MODEL_VERSION_CONTENT, []

    vectors = fit_book_vectors(books)
    als = fit_als(weights)  # None below the ALS minimums, or when implicit is missing or fails
    # Where each book sits when the ids are put in ascending order: the tie-break of equal scores.
    id_rank = np.empty(len(book_ids), dtype=np.int64)
    id_rank[sorted(range(len(book_ids)), key=book_ids.__getitem__)] = np.arange(len(book_ids))

    users: list[UserRecommendations] = []
    for start in range(0, len(user_ids), USER_CHUNK):
        chunk = weights[start : start + USER_CHUNK]
        held = chunk.nonzero()  # (rows, columns) of the books each student already has
        content = _candidate_scores(content_scores(vectors, chunk), held)
        if als is None:
            collaborative, final = None, content
        else:
            collaborative = _candidate_scores(als.scores(start, start + chunk.shape[0]), held)
            # a student's distinct books are the stored entries of their CSR row
            final = blend_scores(content, collaborative, np.diff(chunk.indptr))
        for offset in range(final.shape[0]):
            results = _top_results(
                final[offset],
                content[offset],
                None if collaborative is None else collaborative[offset],
                id_rank,
                book_ids,
                limit,
            )
            if results:
                users.append(UserRecommendations(user_id=user_ids[start + offset], results=results))
    return (MODEL_VERSION_CONTENT if als is None else MODEL_VERSION_HYBRID), users


def collaborative_weight(n: int) -> float:
    """The share of the collaborative part for a student with `n` distinct books: `min(0.7, 0.1 * n)`."""
    return min(MAX_COLLABORATIVE_WEIGHT, COLLABORATIVE_WEIGHT_PER_BOOK * n)


def blend_scores(content: np.ndarray, collaborative: np.ndarray, distinct_books: np.ndarray) -> np.ndarray:
    """`(1 - w) * content + w * collaborative` per student, with `w = collaborative_weight(n)`.

    Both parts are students x books, already scaled to [0, 1]; `distinct_books` holds each student's `n`.
    A convex combination, so the result stays in [0, 1].
    """
    w = np.array([collaborative_weight(int(n)) for n in distinct_books]).reshape(-1, 1)
    return (1.0 - w) * content + w * collaborative


def _weight_matrix(
    book_index: dict[str, int], interactions: Sequence[InteractionIn]
) -> tuple[list[str], csr_matrix]:
    """Students (sorted by id) and their summed, capped weights as a students x books CSR matrix.

    Interactions on books outside the catalog are dropped. A repeated (student, book) pair is summed
    and then capped at MAX_PAIR_WEIGHT.
    """
    rows: list[str] = []
    columns: list[int] = []
    values: list[float] = []
    for interaction in interactions:
        column = book_index.get(interaction.book_id)
        if column is not None:
            rows.append(interaction.user_id)
            columns.append(column)
            values.append(interaction.weight)
    user_ids = sorted(set(rows))
    user_index = {user_id: i for i, user_id in enumerate(user_ids)}
    matrix = coo_matrix(
        (values, ([user_index[user_id] for user_id in rows], columns)),
        shape=(len(user_ids), len(book_index)),
        dtype=np.float64,
    ).tocsr()  # converting to CSR sums the repeated pairs
    np.minimum(matrix.data, MAX_PAIR_WEIGHT, out=matrix.data)
    return user_ids, matrix


def _candidate_scores(scores: np.ndarray, held: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Zero the books each student already has, then scale every row so its best candidate scores 1."""
    scores[held] = 0.0  # never recommend what the student already has
    return _scale_rows(scores)


def _scale_rows(scores: np.ndarray) -> np.ndarray:
    """Divide each row by its maximum, so each student's best candidate scores 1 (a row of 0s stays 0)."""
    peaks = scores.max(axis=1, keepdims=True)
    return np.divide(scores, peaks, out=np.zeros_like(scores), where=peaks > 0)


def _top_results(
    final: np.ndarray,
    content: np.ndarray,
    collaborative: np.ndarray | None,
    id_rank: np.ndarray,
    book_ids: list[str],
    limit: int,
) -> list[RecommendedBook]:
    """The `limit` best books of one student's blended row: score 0 dropped, ties by book id ascending.

    A book is `hybrid` when both parts score above 0, `collaborative` when only the collaborative part
    does, and `content` otherwise (the only label when the collaborative part did not run). Every
    student in the matrix has at least one book, so their collaborative weight is above 0 whenever it ran.
    """
    candidates = np.flatnonzero(final > 0)
    best_first = np.lexsort((id_rank[candidates], -final[candidates]))[:limit]
    return [
        RecommendedBook(
            book_id=book_ids[j],
            score=float(final[j]),
            method=_method(content[j] > 0, collaborative is not None and collaborative[j] > 0),
        )
        for j in candidates[best_first]
    ]


def _method(has_content: bool, has_collaborative: bool) -> RecommendationMethod:
    if has_content and has_collaborative:
        return "hybrid"
    return "collaborative" if has_collaborative else "content"
