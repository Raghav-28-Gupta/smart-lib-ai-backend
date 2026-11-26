"""The content engine of the batch recommender (spec 2026-11-25-book-recommender-design, section 1).

Everything here is pure Python on small hand-made catalogs: no network, no app, no randomness.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from app.ml.recommend.content import content_scores, fit_book_vectors
from app.ml.recommend.engine import recommend_batch
from app.schemas.recommend import BookIn, InteractionIn, UserRecommendations


def book(
    book_id: str,
    title: str,
    author: str = "",
    genre: str | None = None,
    description: str | None = None,
) -> BookIn:
    return BookIn(id=book_id, title=title, author=author, genre=genre, description=description)


def hold(user_id: str, book_id: str, weight: float = 1.0) -> InteractionIn:
    return InteractionIn(user_id=user_id, book_id=book_id, weight=weight)


def catalog() -> list[BookIn]:
    """Two algorithm books and one networking book, as the spec's planted example."""
    return [
        book(
            "a1",
            "Introduction to Algorithms",
            "Rivera",
            "Algorithms",
            "Sorting, graphs and dynamic programming explained with proofs.",
        ),
        book(
            "a2",
            "Data Structures and Algorithms in Java",
            "Okafor",
            "Data Structures",
            "Trees, graphs, hash tables and the algorithms that use them.",
        ),
        book(
            "n1",
            "Computer Networks",
            "Lindqvist",
            "Networks",
            "Layers, protocols and routing from the wire up.",
        ),
    ]


def for_user(users: list[UserRecommendations], user_id: str) -> UserRecommendations:
    return next(user for user in users if user.user_id == user_id)


def ids(user: UserRecommendations) -> list[str]:
    return [result.book_id for result in user.results]


def test_content_ranks_the_same_topic_book_above_the_other_topic() -> None:
    model_version, users = recommend_batch(catalog(), [hold("u1", "a1", 1)], limit=10)

    assert model_version == "tfidf-1"
    results = for_user(users, "u1").results
    assert results[0].book_id == "a2"
    assert results[0].method == "content"
    assert 0 < results[0].score <= 1
    # The networking book shares no term with the algorithms book, so its score is 0 and it is dropped.
    assert ids(for_user(users, "u1")) == ["a2"]


def test_the_best_candidate_scores_one_because_scores_are_scaled_per_student() -> None:
    _, users = recommend_batch(catalog(), [hold("u1", "a1", 1)], limit=10)

    assert for_user(users, "u1").results[0].score == pytest.approx(1.0)


def test_a_student_without_interactions_is_absent() -> None:
    _, users = recommend_batch(catalog(), [], limit=10)
    assert users == []

    # A student whose only interactions name books outside the catalog has no profile either.
    _, users = recommend_batch(catalog(), [hold("u1", "a1"), hold("ghost", "zz-unknown")], limit=10)
    assert [user.user_id for user in users] == ["u1"]


def test_books_the_student_already_has_are_never_recommended() -> None:
    books = [
        book("k1", "Graph Algorithms", "Rivera", "Algorithms", "Graphs and shortest paths."),
        book("k2", "Graph Theory", "Okafor", "Algorithms", "Graphs, trees and shortest paths."),
        book("k3", "Algorithms Unlocked", "Lindqvist", "Algorithms", "Graphs and sorting."),
        book("k4", "Algorithms Illustrated", "Park", "Algorithms", "Graphs and searching."),
        book("k5", "Baking Bread", "Hale", "Cooking", "Flour, water, salt and yeast."),
    ]
    held = {"k1", "k2"}

    _, users = recommend_batch(books, [hold("u1", "k1"), hold("u1", "k2", 0.5)], limit=10)

    recommended = ids(for_user(users, "u1"))
    assert recommended  # the other graph books do show up
    assert not held & set(recommended)
    assert set(recommended) == {"k3", "k4"}  # the baking book shares nothing and scores 0


def test_an_interaction_for_an_unknown_book_is_ignored() -> None:
    with_noise = [hold("u1", "a1", 1), hold("u1", "no-such-book", 3), hold("u2", "no-such-book", 1)]
    plain = [hold("u1", "a1", 1)]

    assert recommend_batch(catalog(), with_noise, limit=10) == recommend_batch(
        catalog(), plain, limit=10
    )


def _mixed_catalog() -> list[BookIn]:
    """Two topics with two books each: how a student's weight splits between them shows in the scores."""
    return [
        book("a1", "Algorithms", "", "Algorithms", "Sorting and graphs."),
        book("a2", "Advanced Algorithms", "", "Algorithms", "Sorting and graphs again."),
        book("n1", "Networks", "", "Networks", "Routing and protocols."),
        book("n2", "Advanced Networks", "", "Networks", "Routing and protocols again."),
    ]


def test_duplicate_pairs_are_summed_and_capped_at_three() -> None:
    books = _mixed_catalog()

    def answer(interactions: list[InteractionIn]) -> list[tuple[str, float]]:
        _, users = recommend_batch(books, interactions, limit=10)
        return [(result.book_id, result.score) for result in for_user(users, "u1").results]

    capped = answer([hold("u1", "a1", 2), hold("u1", "a1", 2), hold("u1", "n1", 1)])
    exactly_three = answer([hold("u1", "a1", 3), hold("u1", "n1", 1)])
    summed_below_cap = answer([hold("u1", "a1", 1), hold("u1", "a1", 1), hold("u1", "n1", 1)])
    single_two = answer([hold("u1", "a1", 2), hold("u1", "n1", 1)])

    assert capped == exactly_three  # 2 + 2 = 4, capped to 3
    assert summed_below_cap == single_two  # 1 + 1 = 2, summed
    assert capped != summed_below_cap  # and the cap is a real limit: 3 is not 2
    # With weight 3 on algorithms against 1 on networks, the algorithms book outranks the networks one.
    assert [book_id for book_id, _ in capped] == ["a2", "n2"]


def test_a_tiny_weight_is_a_weak_signal_not_a_missing_one() -> None:
    # 0 < weight is all the contract asks; the squares of such weights underflow unless the profile is
    # rescaled first, which would silently turn the student into one with no history.
    weak = recommend_batch(catalog(), [hold("u1", "a1", 1e-200)], limit=10)[1]
    plain = recommend_batch(catalog(), [hold("u1", "a1", 1)], limit=10)[1]

    assert ids(for_user(weak, "u1")) == ids(for_user(plain, "u1")) == ["a2"]
    assert for_user(weak, "u1").results[0].score == pytest.approx(for_user(plain, "u1").results[0].score)


def test_all_empty_texts_give_no_results_and_no_error() -> None:
    # Every title is made of stop words only and there is no other text: the vocabulary is empty.
    books = [book(f"e{i}", "the and of to") for i in range(4)]
    interactions = [hold("u1", "e0"), hold("u1", "e1", 2), hold("u2", "e2")]

    model_version, users = recommend_batch(books, interactions, limit=10)

    assert model_version == "tfidf-1"
    assert users == []


def test_limit_is_respected_and_ties_break_on_book_id() -> None:
    same = "Quantum gardening handbook"
    # Listed out of id order on purpose; c1..c5 have identical text, so identical scores.
    books = [
        book("seed", same, "Park", "Botany", "Soil."),
        book("c3", same, "Park", "Botany", "Soil."),
        book("c1", same, "Park", "Botany", "Soil."),
        book("c5", same, "Park", "Botany", "Soil."),
        book("c2", same, "Park", "Botany", "Soil."),
        book("c4", same, "Park", "Botany", "Soil."),
    ]

    _, users = recommend_batch(books, [hold("u1", "seed")], limit=3)
    assert ids(for_user(users, "u1")) == ["c1", "c2", "c3"]
    assert {result.score for result in for_user(users, "u1").results} == {1.0}

    _, users = recommend_batch(books, [hold("u1", "seed")], limit=1)
    assert ids(for_user(users, "u1")) == ["c1"]

    _, users = recommend_batch(books, [hold("u1", "seed")], limit=50)
    assert ids(for_user(users, "u1")) == ["c1", "c2", "c3", "c4", "c5"]


def test_the_same_payload_gives_the_same_answer_twice() -> None:
    interactions = [hold("u2", "n1", 1), hold("u1", "a1", 1.5), hold("u1", "n1", 0.5)]

    first = recommend_batch(catalog(), interactions, limit=10)
    second = recommend_batch(catalog(), interactions, limit=10)

    assert first == second
    assert [user.model_dump_json() for user in first[1]] == [user.model_dump_json() for user in second[1]]


def test_users_come_back_in_user_id_order_whatever_the_input_order() -> None:
    interactions = [hold("u3", "a1"), hold("u1", "n1"), hold("u2", "a1"), hold("u10", "a2")]

    _, users = recommend_batch(catalog(), interactions, limit=10)
    _, shuffled = recommend_batch(catalog(), list(reversed(interactions)), limit=10)

    assert [user.user_id for user in users] == sorted(user.user_id for user in users)
    assert users == shuffled


def test_genre_counts_more_than_description() -> None:
    books = [
        book("seed", "Field notes", "", "Entomology", ""),
        book("by-genre", "Field guide", "", "Entomology", ""),
        book("by-description", "Field guide", "", "Guides", "Entomology"),
    ]

    _, users = recommend_batch(books, [hold("u1", "seed")], limit=10)

    assert ids(for_user(users, "u1"))[0] == "by-genre"


def test_fit_book_vectors_rows_are_unit_length_and_the_matrix_is_sparse() -> None:
    vectors = fit_book_vectors(catalog())

    assert isinstance(vectors, csr_matrix)
    assert vectors.shape[0] == 3
    norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
    assert norms == pytest.approx([1.0, 1.0, 1.0])


def test_fit_book_vectors_gives_a_zero_width_matrix_when_nothing_can_be_a_term() -> None:
    vectors = fit_book_vectors([book("e1", "the of and"), book("e2", ""), book("e3", "a")])

    assert vectors.shape == (3, 0)


def test_content_scores_are_cosines_and_an_empty_profile_scores_zero() -> None:
    vectors = fit_book_vectors(catalog())
    weights = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],  # one book
                [0.0, 0.0, 0.0],  # no interactions
                [3.0, 0.0, 0.0],  # the same book, three times as strongly: the same profile direction
                [1.0, 0.0, 1.0],  # the algorithms and the networking book, which share no term
            ]
        )
    )

    scores = content_scores(vectors, weights)

    assert scores.shape == (4, 3)
    assert scores[0, 0] == pytest.approx(1.0)  # a profile of one book is that book
    assert 0 < scores[0, 1] < 1  # the other algorithms book
    assert scores[0, 2] == pytest.approx(0.0)  # the networking book
    assert not scores[1].any()  # no interactions, no profile, no score
    assert scores[2] == pytest.approx(scores[0])  # cosine ignores the profile's length
    assert scores[3, 0] == pytest.approx(2**-0.5)  # an even mix is at 45 degrees to each of its books
    assert scores[3, 2] == pytest.approx(2**-0.5)


def test_content_scores_do_not_depend_on_how_users_are_chunked() -> None:
    vectors = fit_book_vectors(catalog())
    weights = csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.0, 3.0]]))

    whole = content_scores(vectors, weights)
    parts = np.vstack([content_scores(vectors, weights[i : i + 1]) for i in range(3)])

    assert parts == pytest.approx(whole)


def test_content_scores_with_a_zero_width_vocabulary_are_all_zero() -> None:
    vectors = fit_book_vectors([book("e1", "the of"), book("e2", "and to")])
    weights = csr_matrix(np.array([[1.0, 1.0]]))

    scores = content_scores(vectors, weights)

    assert scores.shape == (1, 2)
    assert not scores.any()
