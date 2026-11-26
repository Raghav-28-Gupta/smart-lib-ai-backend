"""The collaborative part of the batch recommender: implicit ALS fitted from the payload alone.

Stateless, like the content part: nothing is trained ahead of time or stored. Every call factorises
the students-by-books matrix it is given, with a fixed seed, so the same payload gives the same factors
(spec 2026-11-25-book-recommender-design, section 1).
"""

import logging
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)

# Fewer students or interactions than this and there is nothing to learn from: content only.
# "Interactions" are the distinct (student, book) pairs of the matrix, since that is all it holds.
MIN_ALS_USERS = 5
MIN_ALS_INTERACTIONS = 20

# The weights of the matrix (0.5 to 3) are multiplied by this before the fit. implicit reads a stored
# value as the confidence that the student liked the book and gives every unstored pair a confidence of
# 1, so without scaling a single loan (weight 1) would count no more than a book never touched. 40 is the
# value the original implicit-feedback ALS paper (Hu, Koren, Volinsky, 2008) reports using. Nothing is
# tuned. On the planted two-group test data, for random_state=42, each of 1, 3, 5, 10, 15, 20 and 40 ranked
# the unseen same-group book first for every student and 100 did not (at 40 the planted pick leads the
# runner-up by at least 0.25 for every student, in the units of the ENGINE'S ANSWER, the blended score
# scaled to [0, 1]; the raw ALS scores are about fifteen times smaller). That is an observation for this seed, not a property of
# the configuration: under other seeds some students miss their book (see the docstring of
# tests/test_recommend_collaborative.py). After an `implicit` upgrade or a parameter change, re-check it;
# do not re-seed to make it pass.
ALS_CONFIDENCE = 40.0


class AlsFactors(NamedTuple):
    """What the fit leaves behind: one latent vector per student and per book.

    They are float64 copies of implicit's float32 factors, so that a score depends far less on how many
    students are multiplied at once: scores for two block sizes differ by about 1e-16 in float64 (float32
    gave 2e-7 to 2e-6 in two probes), not by nothing. Ids, order and methods were identical for every
    block size probed, but near-tied scores could in principle order differently.
    """

    users: np.ndarray
    books: np.ndarray

    def scores(self, start: int, stop: int) -> np.ndarray:
        """Predicted preference of students `start` to `stop - 1` for every book, clipped below at 0.

        A bounded block of students at a time: the caller asks for one chunk, never for all students.
        """
        block = self.users[start:stop] @ self.books.T
        np.maximum(block, 0.0, out=block)
        return block


def fit_als(weights: csr_matrix) -> AlsFactors | None:
    """Fit ALS on the students-by-books `weights` (CSR, positive values), or None when it cannot run.

    None means "answer with content only": fewer than MIN_ALS_USERS students or MIN_ALS_INTERACTIONS
    pairs, `implicit` not importable, or a fit that failed (the failure is logged by its type alone, so
    no id or number from the payload reaches a log). `implicit` is imported here, not at the top, so a
    host without it still serves the content part.
    """
    if weights.shape[0] < MIN_ALS_USERS or weights.nnz < MIN_ALS_INTERACTIONS:
        return None
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        logger.warning("implicit is not installed; answering with content picks only")
        return None
    try:
        # BLAS on one thread: implicit warns, and runs up to 10x slower, when BLAS has its own thread pool.
        # The model is built inside the block because implicit checks the BLAS configuration there.
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(
                factors=16, regularization=0.1, iterations=15, random_state=42, use_gpu=False
            )
            model.fit(weights * ALS_CONFIDENCE, show_progress=False)
        return AlsFactors(
            np.asarray(model.user_factors, dtype=np.float64), np.asarray(model.item_factors, dtype=np.float64)
        )
    except Exception as error:
        logger.warning("ALS fit failed (%s); answering with content picks only", type(error).__name__)
        return None


def als_scores(weights: csr_matrix) -> np.ndarray | None:
    """Every student's ALS score for every book (students by books, float64, clipped at 0), or None.

    The whole matrix at once, for callers that want it and for the tests. The engine does not use it: it
    fits once with `fit_als` and scores a chunk of students at a time, so memory stays bounded however
    many students there are (the whole matrix is students x books floats: gigabytes at the contract's caps).
    """
    factors = fit_als(weights)
    return None if factors is None else factors.scores(0, weights.shape[0])
