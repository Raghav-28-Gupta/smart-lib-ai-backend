"""The content-based part of the batch recommender: TF-IDF book vectors and profile similarity.

Stateless, like everything here: nothing is trained ahead of time or stored. Every call fits the
vocabulary from the books it is given (spec 2026-11-25-book-recommender-design, section 1).
"""

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import TfidfVectorizer

from app.schemas.recommend import BookIn


def book_text(book: BookIn) -> str:
    """One text per book: title, author, genre twice (so it counts more) and description."""
    genre = book.genre or ""
    return " ".join((book.title, book.author, genre, genre, book.description or ""))


def fit_book_vectors(books: Sequence[BookIn]) -> csr_matrix:
    """The TF-IDF vector of every book, one L2-normalised row each, in the order of `books`.

    English stop words, sublinear term frequency. When no book has a single usable term (all texts
    empty, or made only of stop words) the vocabulary is empty and the result has shape
    `(len(books), 0)`: every similarity against it is 0, and nothing raises.
    """
    texts = [book_text(book) for book in books]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(text) for text in texts):
        return csr_matrix((len(texts), 0), dtype=np.float64)
    return vectorizer.fit_transform(texts).tocsr()


def content_scores(book_vectors: csr_matrix, weights: csr_matrix) -> np.ndarray:
    """The cosine similarity of each user's weighted profile with every book, as a dense matrix.

    `weights` is users by books (any non-negative weights; here the summed interaction weights).
    A user's profile is the weighted sum of the vectors of their books; the rows of `book_vectors`
    are already unit length, so the dot product with the unit-length profile is the cosine. A user with
    an empty profile (no weights, or only books without a usable term) scores 0 on every book.

    The result holds users x books floats: the caller passes a bounded chunk of users at a time.
    """
    n_users, n_books = weights.shape[0], book_vectors.shape[0]
    if book_vectors.shape[1] == 0 or weights.nnz == 0:
        return np.zeros((n_users, n_books), dtype=np.float64)
    # A cosine ignores the profile's length, so each user's weights are first scaled to a largest of 1:
    # the squares below would underflow to 0 for a weight like 1e-200, and a weak signal would vanish.
    heaviest = weights.max(axis=1).toarray().ravel()
    unit_weights = diags(np.divide(1.0, heaviest, out=np.zeros_like(heaviest), where=heaviest > 0)) @ weights
    profiles = unit_weights @ book_vectors
    lengths = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
    inverse = np.divide(1.0, lengths, out=np.zeros_like(lengths), where=lengths > 0)
    unit_profiles = diags(inverse) @ profiles
    return (unit_profiles @ book_vectors.T).toarray()
