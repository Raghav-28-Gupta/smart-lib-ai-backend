"""The collaborative part of the batch recommender and the blend (spec 2026-11-25-book-recommender-design).

The planted structure: 40 students in two groups of 20. Group X reads the books x1..x5 and group Y
the books y1..y5; no two books share a word (unless a test adds one on purpose), so the content part
sees nothing and only the collaborative part can tell the groups apart. Student `i` of a group borrows
four of its five books (the one left out is book number `i % 5`), so a student's unseen same-group book
is the one to find. No real-world quality is claimed: these tests prove the plumbing and the planted
structure.

How much of the planted structure ALS recovers depends on its random initialisation. The engine fits with
`random_state=42`. There every one of the 40 students gets their unseen same-group book first, and it leads
the runner-up by at least 0.25 (in the answer's scores, scaled to [0, 1]). That is an observation for that seed, not a property of the configuration:
replacing the seed in memory (nothing edited) gave, across seeds 0 to 39, between 28 and 40 of the 40
students finding it (12 missed at the worst, 2 to 3 on average), and across seeds 0 to 199 as few as 16 (five
seeds in 200 fell below two thirds, and only one below half: seed 138 with 16). So the tests that cover every
student only assert that at least half of them find it (`MOST_FIND_THEIR_BOOK`; by chance only a handful
of 40 would), and the tests that are about one student (x4) carry `seed_42`
in their names.
If `implicit` is upgraded, or an ALS parameter is changed, and a seed-42 test goes red, re-check the claim
by hand (see how far it moved) before touching it. Choosing another seed to make it pass again would only
hide the change.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from threadpoolctl import threadpool_info

from app.ml.recommend import engine
from app.ml.recommend.collaborative import (
    ALS_CONFIDENCE,
    MIN_ALS_INTERACTIONS,
    MIN_ALS_USERS,
    AlsFactors,
    als_scores,
)
from app.ml.recommend.engine import (
    MAX_COLLABORATIVE_WEIGHT,
    blend_scores,
    collaborative_weight,
    recommend_batch,
)
from app.schemas.recommend import BatchRequest, BookIn, InteractionIn, UserRecommendations

FIXTURE = Path(__file__).parent / "fixtures" / "recommend_batch_example.json"

GROUPS = "xy"
GROUP_SIZE = 20
BOOKS_PER_GROUP = 5
MOST_FIND_THEIR_BOOK = 20  # half of the 40 students (chance alone gives a handful); across seeds 0 to 199 only seed 138 (16) is lower
_SYLLABLES = ("ba", "do", "ki", "lu", "me", "no", "pa", "ri")


def made_up_word(n: int) -> str:
    """A word nobody has: distinct for every n below 64 and never an English stop word."""
    return _SYLLABLES[n % 8] + _SYLLABLES[n // 8 % 8] + "qu"


def planted_books(shared: dict[str, str] | None = None) -> list[BookIn]:
    """x1..x5 and y1..y5, each with four words of its own (title, author, genre, description).

    `shared` maps a group letter to one word that every book of that group also carries.
    """
    shared = shared or {}
    books = []
    for group_number, group in enumerate(GROUPS):
        for k in range(BOOKS_PER_GROUP):
            first = (group_number * BOOKS_PER_GROUP + k) * 4
            extra = f" {shared[group]}" if group in shared else ""
            books.append(
                BookIn(
                    id=f"{group}{k + 1}",
                    title=made_up_word(first),
                    author=made_up_word(first + 1),
                    genre=made_up_word(first + 2),
                    description=made_up_word(first + 3) + extra,
                )
            )
    return books


def student_id(group: str, i: int) -> str:
    return f"s{GROUPS.index(group) * GROUP_SIZE + i:02d}"  # zero-padded: id order is student order


def planted_interactions() -> list[InteractionIn]:
    """Student `i` of each group holds four of the group's five books; book `i % 5` is left out."""
    return [
        InteractionIn(user_id=student_id(group, i), book_id=f"{group}{k + 1}", weight=1)
        for group in GROUPS
        for i in range(GROUP_SIZE)
        for k in range(BOOKS_PER_GROUP)
        if k != i % BOOKS_PER_GROUP
    ]


def for_user(users: list[UserRecommendations], user_id: str) -> UserRecommendations:
    return next(user for user in users if user.user_id == user_id)


def ids(user: UserRecommendations) -> list[str]:
    return [result.book_id for result in user.results]


def methods(user: UserRecommendations) -> dict[str, str]:
    return {result.book_id: result.method for result in user.results}


def assert_same_answer(
    actual: tuple[str, list[UserRecommendations]], expected: tuple[str, list[UserRecommendations]]
) -> None:
    """Model version, users, book ids, order and methods exactly; scores to a rounding error."""
    assert actual[0] == expected[0]
    assert [user.user_id for user in actual[1]] == [user.user_id for user in expected[1]]
    for got, want in zip(actual[1], expected[1], strict=True):
        assert ids(got) == ids(want)
        assert [r.method for r in got.results] == [r.method for r in want.results]
        assert [r.score for r in got.results] == pytest.approx([r.score for r in want.results])


def matrix(users: int, books: int, per_user: int) -> csr_matrix:
    """Each user holds `per_user` consecutive books (wrapping), weight 1."""
    dense = np.zeros((users, books))
    for u in range(users):
        for k in range(per_user):
            dense[u, (u + k) % books] = 1.0
    return csr_matrix(dense)


def test_the_made_up_words_are_distinct_and_not_stop_words() -> None:
    words = [made_up_word(n) for n in range(40)]
    assert len(set(words)) == 40
    assert not set(words) & ENGLISH_STOP_WORDS


# --- when the collaborative part runs ---


def test_below_five_users_or_twenty_interactions_is_content_only() -> None:
    # Four students of group X: 16 interactions, below both minimums.
    four = [i for i in planted_interactions() if i.user_id in {student_id("x", n) for n in range(4)}]
    assert len(four) == 16

    model_version, users = recommend_batch(planted_books({"x": "alphaqu"}), four, limit=10)

    assert model_version == "tfidf-1"
    assert users  # the shared word still gives content picks
    assert {r.method for user in users for r in user.results} == {"content"}


def test_the_minimums_are_five_users_and_twenty_interactions() -> None:
    assert (MIN_ALS_USERS, MIN_ALS_INTERACTIONS) == (5, 20)
    assert als_scores(matrix(users=4, books=10, per_user=10)) is None  # 40 interactions, 4 users
    assert als_scores(matrix(users=5, books=10, per_user=3)) is None  # 15 interactions
    nineteen = matrix(users=5, books=10, per_user=4)
    nineteen[0, 3] = 0  # 20 interactions become 19
    nineteen.eliminate_zeros()
    assert nineteen.nnz == 19
    assert als_scores(nineteen) is None
    scores = als_scores(matrix(users=5, books=10, per_user=4))  # exactly 5 users, 20 interactions
    assert scores is not None
    assert scores.shape == (5, 10)
    assert (scores >= 0).all()  # clipped below 0


def test_the_collaborative_part_runs_with_enough_history() -> None:
    model_version, users = recommend_batch(planted_books(), planted_interactions(), limit=10)

    assert model_version == "tfidf-als-1"
    assert len(users) == 40
    assert {r.method for user in users for r in user.results} == {"collaborative"}  # content sees nothing


def test_seed_42_the_unseen_same_group_book_of_student_x4_is_their_first_collaborative_pick() -> None:
    """A seed-42 observation about one student, not a property of the configuration (module docstring)."""
    _, users = recommend_batch(planted_books(), planted_interactions(), limit=10)

    student = for_user(users, student_id("x", 4))  # holds x1..x4, so x5 is the one to find
    ranked = ids(student)
    assert "x5" in ranked[:2]
    assert methods(student)["x5"] == "collaborative"  # its content score is 0
    # The brief also asks for no y book in the top 3, which cannot hold: x5 is the only unseen X book,
    # so every other candidate is a y book. What does hold is that no y book outranks x5.
    assert ranked[0] == "x5"
    best_y = max(r.score for r in student.results if r.book_id.startswith("y"))
    assert best_y < student.results[0].score


def test_most_students_are_pointed_at_the_book_their_group_has_and_they_do_not() -> None:
    """The property that holds across ALS seeds: at least half of the 40 students find their book.

    Which students miss it changes with the seed (across seeds 0 to 39 at most 12 of 40 did), so this
    asserts a share, not every student. What does not depend on the seed is asserted for all 40: a
    student is never offered a book they already hold, and a pick found by ALS alone is `collaborative`.
    """
    _, users = recommend_batch(planted_books(), planted_interactions(), limit=10)

    found = 0
    for group in GROUPS:
        for i in range(GROUP_SIZE):
            student = for_user(users, student_id(group, i))
            held = {f"{group}{k + 1}" for k in range(BOOKS_PER_GROUP) if k != i % BOOKS_PER_GROUP}
            assert not held & set(ids(student)), student.user_id
            assert set(methods(student).values()) == {"collaborative"}, student.user_id  # content sees nothing
            if ids(student)[0] == f"{group}{i % BOOKS_PER_GROUP + 1}":
                found += 1

    assert found >= MOST_FIND_THEIR_BOOK, f"only {found} of {len(GROUPS) * GROUP_SIZE} found their book first"


def test_seed_42_a_shared_word_makes_the_pick_of_student_x4_hybrid() -> None:
    """A seed-42 observation about one student, not a property of the configuration (module docstring)."""
    books = planted_books({"x": "sharedqu"})

    _, users = recommend_batch(books, planted_interactions(), limit=10)

    student = for_user(users, student_id("x", 4))
    assert ids(student)[0] == "x5"
    assert methods(student)["x5"] == "hybrid"  # content (the shared word) and collaborative both above 0
    assert all(0 < r.score <= 1 for user in users for r in user.results)
    assert student.results[0].score == pytest.approx(1.0)  # both parts peak on x5: (1 - w) + w


# --- the blend weight ---


def test_collaborative_weight_formula() -> None:
    assert MAX_COLLABORATIVE_WEIGHT == 0.7
    assert collaborative_weight(0) == 0
    assert collaborative_weight(1) == pytest.approx(0.1)
    assert collaborative_weight(3) == pytest.approx(0.3)
    assert collaborative_weight(6) == pytest.approx(0.6)
    assert collaborative_weight(7) == pytest.approx(0.7)
    assert collaborative_weight(10) == pytest.approx(0.7)
    assert collaborative_weight(10_000) == pytest.approx(0.7)


def test_blend_scores_is_one_minus_w_content_plus_w_collaborative_per_student() -> None:
    content = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
    collaborative = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5]])
    distinct_books = np.array([1, 7])  # w = 0.1 for the first student, 0.7 for the second

    blended = blend_scores(content, collaborative, distinct_books)

    assert blended == pytest.approx(
        np.array(
            [
                [0.9 * 1.0, 0.9 * 0.5 + 0.1 * 1.0, 0.1 * 1.0],
                [0.7 * 1.0, 0.3 * 1.0, 0.7 * 0.5],
            ]
        )
    )


def test_a_blended_score_never_leaves_zero_to_one() -> None:
    ones = np.ones((11, 1))  # a candidate that is the best on both parts
    distinct_books = np.arange(0, 11)

    blended = blend_scores(ones, ones, distinct_books)

    assert ((blended > 0) & (blended <= 1.0)).all()


def test_seed_42_the_method_follows_which_parts_are_above_zero() -> None:
    # A seed-42 observation about one student (module docstring): it needs ALS to score every book of
    # the other group above 0 for x4, which another seed may not.
    # z1 carries the X group's shared word but nobody has read it: content above 0, collaborative 0.
    z1 = BookIn(id="z1", title=made_up_word(40), author=made_up_word(41), genre=None, description="sharedqu")
    books = [*planted_books({"x": "sharedqu"}), z1]

    model_version, users = recommend_batch(books, planted_interactions(), limit=10)

    assert model_version == "tfidf-als-1"
    student = for_user(users, student_id("x", 4))
    assert methods(student)["x5"] == "hybrid"  # both parts above 0
    assert methods(student)["z1"] == "content"  # content only
    assert all(m == "collaborative" for b, m in methods(student).items() if b.startswith("y"))


def test_each_student_is_blended_with_the_weight_of_their_own_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collaborative part is replaced by known scores, so the blend can be checked to the digit.

    s0 holds one book (w = 0.1) and s1 two (w = 0.2); every book carries one common word, so every
    candidate has a content score above 0 as well. The expected scores are
    `(1 - w) * content + w * collaborative`, the content scores being those of the same payload
    without the collaborative part. A chunk of one student makes the weights come from the right rows.
    """
    books = [
        BookIn(id=f"k{i}", title=made_up_word(i), author="", genre=None, description="sharedqu")
        for i in range(4)
    ]
    interactions = [
        InteractionIn(user_id="s0", book_id="k0", weight=1),
        InteractionIn(user_id="s1", book_id="k0", weight=1),
        InteractionIn(user_id="s1", book_id="k1", weight=1),
    ]
    raw = np.array([[0.0, 0.2, 0.4, 0.1], [0.0, 0.0, 0.1, 0.3]])  # students x books, ALS scores
    content_only = recommend_batch(books, interactions, limit=10)  # fewer than 5 students: no ALS
    assert content_only[0] == "tfidf-1"
    monkeypatch.setattr(engine, "fit_als", lambda weights: AlsFactors(raw, np.eye(4)))

    for chunk in (512, 1):
        monkeypatch.setattr(engine, "USER_CHUNK", chunk)
        model_version, users = recommend_batch(books, interactions, limit=10)

        assert model_version == "tfidf-als-1"
        for user_id, held, w in (("s0", {"k0"}, 0.1), ("s1", {"k0", "k1"}, 0.2)):
            row = raw[int(user_id[1])].copy()
            row[[int(b[1]) for b in held]] = 0.0  # a held book is never a candidate
            collaborative = row / row.max()
            content = {r.book_id: r.score for r in for_user(content_only[1], user_id).results}
            expected = {
                f"k{j}": (1 - w) * content[f"k{j}"] + w * collaborative[j]
                for j in range(4)
                if f"k{j}" not in held
            }
            got = {r.book_id: r.score for r in for_user(users, user_id).results}
            assert got == pytest.approx(expected), (user_id, chunk)
            assert {r.method for r in for_user(users, user_id).results} <= {"hybrid", "collaborative"}


def test_the_blend_weight_counts_distinct_books_not_total_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    """A renewed or repeated loan weighs more, but it is still one book: `n` is the number of books.

    Every student here holds books whose weights add up to more than their number of books, so a
    weight sum in place of the count (the mutant of Task 2's review) gives each the wrong `w`:
    s0 holds 3 books of weight 1.5 (w = 0.3, not 0.4 or 0.45), s1 one book of weight 3 (w = 0.1, not
    0.3) and s2 two books of weight 0.5 (w = 0.2, not 0.1: a sum below the count goes wrong too).
    The collaborative part is replaced by known scores, as in the test above, and the expected scores
    are `(1 - w) * content + w * collaborative` with each w written out, not computed.
    """
    books = [
        BookIn(id=f"k{i}", title=made_up_word(i), author="", genre=None, description="sharedqu")
        for i in range(6)
    ]
    interactions = [
        *(InteractionIn(user_id="s0", book_id=f"k{j}", weight=1.5) for j in (0, 1, 2)),
        InteractionIn(user_id="s1", book_id="k0", weight=3),
        *(InteractionIn(user_id="s2", book_id=f"k{j}", weight=0.5) for j in (3, 4)),
    ]
    held = {"s0": {0, 1, 2}, "s1": {0}, "s2": {3, 4}}
    w = {"s0": 0.3, "s1": 0.1, "s2": 0.2}
    raw = np.array(  # students (s0, s1, s2) by books (k0 to k5), ALS scores before the held books are dropped
        [
            [0.9, 0.9, 0.9, 0.4, 0.1, 0.2],
            [0.9, 0.3, 0.2, 0.5, 0.1, 0.4],
            [0.2, 0.3, 0.1, 0.9, 0.9, 0.5],
        ]
    )
    content_only = recommend_batch(books, interactions, limit=10)  # three students: no ALS
    assert content_only[0] == "tfidf-1"
    monkeypatch.setattr(engine, "fit_als", lambda weights: AlsFactors(raw, np.eye(6)))

    for chunk in (512, 1):
        monkeypatch.setattr(engine, "USER_CHUNK", chunk)
        model_version, users = recommend_batch(books, interactions, limit=10)

        assert model_version == "tfidf-als-1"
        for row, user_id in enumerate(("s0", "s1", "s2")):
            collaborative = raw[row].copy()
            collaborative[list(held[user_id])] = 0.0  # a held book is never a candidate
            collaborative = collaborative / collaborative.max()
            content = {r.book_id: r.score for r in for_user(content_only[1], user_id).results}
            expected = {
                f"k{j}": (1 - w[user_id]) * content[f"k{j}"] + w[user_id] * collaborative[j]
                for j in range(6)
                if j not in held[user_id]
            }
            got = {r.book_id: r.score for r in for_user(users, user_id).results}
            assert got == pytest.approx(expected), (user_id, chunk)


def test_the_als_is_built_and_fitted_as_specified(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen["init"] = kwargs
            seen["blas_threads_at_init"] = blas_threads()

        def fit(self, user_items: csr_matrix, show_progress: bool = True) -> None:
            seen["fit"] = user_items.copy()
            seen["show_progress"] = show_progress
            seen["blas_threads_in_fit"] = blas_threads()
            self.user_factors = np.ones((user_items.shape[0], 2), dtype=np.float32)
            self.item_factors = np.ones((user_items.shape[1], 2), dtype=np.float32)

    def blas_threads() -> list[int]:
        return [info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"]

    monkeypatch.setattr("implicit.als.AlternatingLeastSquares", Recorder)
    weights = matrix(users=6, books=10, per_user=4)

    scores = als_scores(weights)

    assert seen["init"] == {
        "factors": 16,
        "regularization": 0.1,
        "iterations": 15,
        "random_state": 42,
        "use_gpu": False,
    }
    fitted = seen["fit"]
    assert fitted.format == "csr" and fitted.shape == (6, 10)  # students by books, as implicit expects
    assert (fitted != weights * ALS_CONFIDENCE).nnz == 0  # the one fixed confidence scaling
    assert seen["show_progress"] is False
    # BLAS on one thread while the model is built (implicit checks it there) and fitted.
    assert set(seen["blas_threads_at_init"]) <= {1} and set(seen["blas_threads_in_fit"]) <= {1}
    assert seen["blas_threads_in_fit"]  # and a BLAS library was there to limit
    assert scores is not None and scores.shape == (6, 10)


# --- determinism, degradation, the committed contract ---


def test_the_answer_is_identical_across_runs_with_als() -> None:
    books, interactions = planted_books({"x": "sharedqu"}), planted_interactions()

    first = recommend_batch(books, interactions, limit=10)
    second = recommend_batch(books, interactions, limit=10)

    assert first[0] == "tfidf-als-1"
    assert [u.model_dump() for u in first[1]] == [u.model_dump() for u in second[1]]  # exactly equal


def test_a_missing_implicit_degrades_to_content_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "implicit", None)
    monkeypatch.setitem(sys.modules, "implicit.als", None)

    assert als_scores(matrix(users=40, books=10, per_user=4)) is None
    model_version, users = recommend_batch(
        planted_books({"x": "sharedqu"}), planted_interactions(), limit=10
    )

    assert model_version == "tfidf-1"
    assert {r.method for user in users for r in user.results} == {"content"}


def test_a_failing_fit_degrades_to_content_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def __init__(self, **_: object) -> None: ...

        def fit(self, *_: object, **__: object) -> None:
            raise ValueError("NaN encountered in factors")

    monkeypatch.setattr("implicit.als.AlternatingLeastSquares", Broken)

    model_version, users = recommend_batch(
        planted_books({"x": "sharedqu"}), planted_interactions(), limit=10
    )

    assert model_version == "tfidf-1"
    assert users
    assert {r.method for user in users for r in user.results} == {"content"}


def test_the_committed_contract_fixture_is_still_content_only() -> None:
    request = BatchRequest.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8"))["request"])
    # Below the ALS minimums (5 students and 20 interactions): the fixture stays a content-only contract.
    assert len({i.user_id for i in request.interactions}) < MIN_ALS_USERS
    assert len(request.interactions) < MIN_ALS_INTERACTIONS

    model_version, users = recommend_batch(request.books, request.interactions, request.limit)

    assert model_version == "tfidf-1"
    assert users
    assert {r.method for user in users for r in user.results} == {"content"}


# --- chunking ---


def content_only_payload() -> tuple[list[BookIn], list[InteractionIn]]:
    """Seven students and ten interactions: below the ALS minimums, so only the content part runs.

    Each group's books share a word, so a student is offered the group's other books by content.
    """
    books = planted_books({"x": "sharedxqu", "y": "sharedyqu"})
    interactions = []
    for j in range(7):
        interactions.append(InteractionIn(user_id=f"s{j:02d}", book_id=f"x{j % 5 + 1}", weight=1))
        if j % 2:
            interactions.append(InteractionIn(user_id=f"s{j:02d}", book_id=f"y{j % 5 + 1}", weight=2))
    return books, interactions


def als_payload() -> tuple[list[BookIn], list[InteractionIn]]:
    return planted_books({"x": "sharedxqu"}), planted_interactions()


@pytest.mark.parametrize("payload", [content_only_payload, als_payload], ids=["content-only", "als"])
@pytest.mark.parametrize("chunk", [2, 1])
def test_chunking_does_not_change_the_answer(
    monkeypatch: pytest.MonkeyPatch, payload, chunk: int
) -> None:
    books, interactions = payload()
    expected = recommend_batch(books, interactions, limit=10)
    assert expected[0] == ("tfidf-als-1" if payload is als_payload else "tfidf-1")
    assert len(expected[1]) >= 7
    # The students must differ, or a student served another one's picks would go unseen.
    assert len({tuple(ids(user)) for user in expected[1]}) > 1

    monkeypatch.setattr(engine, "USER_CHUNK", chunk)

    assert_same_answer(recommend_batch(books, interactions, limit=10), expected)
