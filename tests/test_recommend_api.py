"""Contract tests for POST /recommend/batch (spec 2026-11-25-book-recommender-design, sections 1 and 2).

The route is stateless: the Node backend sends the catalog and every student's interactions, and
gets the top picks per student back. It is protected by the shared secret `x-ai-token`.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.recommend import BatchRequest, BatchResponse

URL = "/recommend/batch"
FIXTURE = Path(__file__).parent / "fixtures" / "recommend_batch_example.json"


@pytest.fixture
def api() -> Iterator[TestClient]:
    """The real app without its lifespan (the recommender never needs the no-show model)."""
    app.dependency_overrides.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Switch the recommender on with the token `t`; returns the header that matches it."""
    monkeypatch.setenv("AI_TOKEN", "t")
    return {"x-ai-token": "t"}


def make_payload() -> dict[str, Any]:
    """A small valid request: two algorithm books and a networking book, one student."""
    return {
        "books": [
            {
                "id": "a1",
                "title": "Introduction to Algorithms",
                "author": "Rivera",
                "genre": "Algorithms",
                "description": "Sorting, graphs and dynamic programming.",
            },
            {
                "id": "a2",
                "title": "Data Structures and Algorithms in Java",
                "author": "Okafor",
                "genre": "Data Structures",
                "description": "Trees, graphs, hash tables and algorithms.",
            },
            {
                "id": "n1",
                "title": "Computer Networks",
                "author": "Lindqvist",
                "genre": None,
                "description": None,
            },
        ],
        "interactions": [{"user_id": "u1", "book_id": "a1", "weight": 1}],
        "limit": 10,
    }


def tiny_book(i: int) -> dict[str, Any]:
    return {"id": f"b{i}", "title": "t", "author": "a", "genre": None, "description": None}


# --- the shared secret ---


@pytest.mark.parametrize("configured", [None, "", "   "])
@pytest.mark.parametrize("send_header", [False, True])
def test_token_unset_or_blank_answers_404(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    send_header: bool,
) -> None:
    if configured is None:
        monkeypatch.delenv("AI_TOKEN", raising=False)
    else:
        monkeypatch.setenv("AI_TOKEN", configured)
    headers = {"x-ai-token": configured or "anything"} if send_header else {}

    response = api.post(URL, json=make_payload(), headers=headers)

    assert response.status_code == 404
    # The same body as a route that does not exist: the recommender is simply off.
    assert response.json() == api.get("/no-such-route").json()


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-ai-token": ""},
        {"x-ai-token": "wrong"},
        {"x-ai-token": "T"},
        {"x-ai-token": "t "},  # a space is not part of the token it is compared with
        {"x-ai-token": "tt"},
        {"x-ai-token": "Bearer t"},
        {"x-ai-token": "é".encode("latin-1")},  # non-ASCII must be a 401, not a 500
    ],
)
def test_missing_or_wrong_token_answers_401(
    api: TestClient, token: dict[str, str], headers: dict[str, Any]
) -> None:
    response = api.post(URL, json=make_payload(), headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_an_invalid_body_without_the_token_is_401_not_422(
    api: TestClient, token: dict[str, str]
) -> None:
    # The token is checked first: a stranger learns nothing about the contract from a 422.
    response = api.post(URL, json={"nonsense": True})

    assert response.status_code == 401


def test_a_configured_token_with_spaces_around_it_still_matches(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_TOKEN", "  secret-value \n")

    response = api.post(URL, json=make_payload(), headers={"x-ai-token": "secret-value"})

    assert response.status_code == 200


def test_the_token_is_read_on_every_request(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TOKEN", "first")
    assert api.post(URL, json=make_payload(), headers={"x-ai-token": "first"}).status_code == 200

    monkeypatch.setenv("AI_TOKEN", "second")
    assert api.post(URL, json=make_payload(), headers={"x-ai-token": "first"}).status_code == 401
    assert api.post(URL, json=make_payload(), headers={"x-ai-token": "second"}).status_code == 200

    monkeypatch.delenv("AI_TOKEN")
    assert api.post(URL, json=make_payload(), headers={"x-ai-token": "second"}).status_code == 404


def test_health_and_no_show_do_not_need_the_token(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_TOKEN", "t")

    assert api.get("/health").status_code == 200
    # No model is loaded in this client: the answer is the 503 of the route itself, never a 401 or 404.
    no_show = api.post(
        "/predict/no-show",
        json={
            "hour_of_day": 10,
            "day_of_week": 2,
            "duration_minutes": 120,
            "lead_time_hours": 24.0,
            "resource_type": "room",
            "past_attendance_ratio": 0.9,
            "consecutive_attended_count": 5,
            "recent_noshows_count": 0,
            "reliability_score": 80,
            "bookings_last_30d": 4,
        },
    )
    assert no_show.status_code not in (401, 404)


def test_the_token_is_never_in_a_response(
    api: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "s3cr3t-Tok3n-9f2a71"
    wrong = "wr0ng-Tok3n-1c88de"
    monkeypatch.setenv("AI_TOKEN", secret)
    bad_body = {**make_payload(), "limit": 0}
    duplicate_ids = make_payload()
    duplicate_ids["books"].append(dict(duplicate_ids["books"][0]))

    with caplog.at_level(logging.DEBUG):
        responses = [
            api.post(URL, json=make_payload(), headers={"x-ai-token": secret}),  # 200
            api.post(URL, json=make_payload(), headers={"x-ai-token": wrong}),  # 401
            api.post(URL, json=make_payload()),  # 401
            api.post(URL, json=bad_body, headers={"x-ai-token": secret}),  # 422
            api.post(URL, json=duplicate_ids, headers={"x-ai-token": secret}),  # 422
        ]
    monkeypatch.delenv("AI_TOKEN")
    responses.append(api.post(URL, json=make_payload(), headers={"x-ai-token": secret}))  # 404

    assert [response.status_code for response in responses] == [200, 401, 401, 422, 422, 404]
    for response in responses:
        assert secret not in response.text
        assert wrong not in response.text
        assert secret not in json.dumps(dict(response.headers))
    assert secret not in caplog.text
    assert wrong not in caplog.text


# --- the contract ---


def test_right_token_answers_200_with_the_contract_shape(api: TestClient, token: dict[str, str]) -> None:
    response = api.post(URL, json=make_payload(), headers=token)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model_version", "users"}
    assert body["model_version"] == "tfidf-1"
    assert [user["user_id"] for user in body["users"]] == ["u1"]
    user = body["users"][0]
    assert set(user) == {"user_id", "results"}
    assert [result["book_id"] for result in user["results"]] == ["a2"]
    for result in user["results"]:
        assert set(result) == {"book_id", "score", "method"}
        assert 0 <= result["score"] <= 1
        assert result["method"] == "content"
    BatchResponse.model_validate(body)  # and it parses as the published response model


def test_no_interactions_answers_200_with_no_users(api: TestClient, token: dict[str, str]) -> None:
    payload = {**make_payload(), "interactions": []}

    response = api.post(URL, json=payload, headers=token)

    assert response.status_code == 200
    assert response.json() == {"model_version": "tfidf-1", "users": []}


def test_the_limit_defaults_to_ten_and_is_applied(api: TestClient, token: dict[str, str]) -> None:
    books = [
        {"id": f"k{i:02d}", "title": "Gardening handbook", "author": "Park", "genre": "Botany", "description": None}
        for i in range(14)
    ]
    interactions = [{"user_id": "u1", "book_id": "k00", "weight": 1}]

    default = api.post(URL, json={"books": books, "interactions": interactions}, headers=token)
    limited = api.post(URL, json={"books": books, "interactions": interactions, "limit": 4}, headers=token)

    assert [r["book_id"] for r in default.json()["users"][0]["results"]] == [f"k{i:02d}" for i in range(1, 11)]
    assert [r["book_id"] for r in limited.json()["users"][0]["results"]] == [f"k{i:02d}" for i in range(1, 5)]


def test_all_empty_texts_answer_200_with_no_users(api: TestClient, token: dict[str, str]) -> None:
    payload = {
        "books": [{"id": f"e{i}", "title": "the and of", "author": "", "genre": None, "description": None} for i in range(3)],
        "interactions": [{"user_id": "u1", "book_id": "e0", "weight": 1}],
    }

    response = api.post(URL, json=payload, headers=token)

    assert response.status_code == 200
    assert response.json() == {"model_version": "tfidf-1", "users": []}


def test_duplicate_book_ids_are_422(api: TestClient, token: dict[str, str]) -> None:
    payload = make_payload()
    payload["books"].append({**payload["books"][2], "id": "dup-marker-77"})
    payload["books"].append({**payload["books"][2], "id": "dup-marker-77"})

    response = api.post(URL, json=payload, headers=token)

    assert response.status_code == 422
    assert "dup-marker-77" not in response.text


@pytest.mark.parametrize("weight", [0, 0.0, -1, -0.001, 3.0001, 4, 1000, "heavy", None])
def test_zero_negative_or_over_three_weight_is_422(
    api: TestClient, token: dict[str, str], weight: Any
) -> None:
    payload = make_payload()
    payload["interactions"][0]["weight"] = weight

    response = api.post(URL, json=payload, headers=token)

    assert response.status_code == 422


def test_a_nan_or_infinite_weight_is_422_not_a_500(api: TestClient, token: dict[str, str]) -> None:
    for literal in ("NaN", "Infinity", "-Infinity"):
        body = json.dumps(make_payload()).replace('"weight": 1', f'"weight": {literal}')
        response = api.post(URL, content=body, headers={**token, "content-type": "application/json"})
        assert response.status_code == 422, literal


@pytest.mark.parametrize("weight", [3, 3.0, 0.5, 0.0001])
def test_weights_inside_the_bounds_are_accepted(api: TestClient, token: dict[str, str], weight: float) -> None:
    payload = make_payload()
    payload["interactions"][0]["weight"] = weight

    assert api.post(URL, json=payload, headers=token).status_code == 200


def test_an_unknown_field_is_422_and_the_input_is_not_echoed(
    api: TestClient, token: dict[str, str]
) -> None:
    echo = "needle-do-not-echo-4471"
    top = {**make_payload(), "surprise": echo}
    in_book = make_payload()
    in_book["books"][0]["isbn"] = echo
    in_interaction = make_payload()
    in_interaction["interactions"][0]["email"] = echo
    wrong_type = {**make_payload(), "limit": echo}
    bad_weight = make_payload()
    bad_weight["interactions"][0]["weight"] = echo

    for payload in (top, in_book, in_interaction, wrong_type, bad_weight):
        response = api.post(URL, json=payload, headers=token)

        assert response.status_code == 422
        assert echo not in response.text
        for problem in response.json()["detail"]:
            assert set(problem) == {"type", "loc", "msg"}


def test_a_book_with_a_missing_required_field_is_422(api: TestClient, token: dict[str, str]) -> None:
    payload = make_payload()
    del payload["books"][0]["title"]

    assert api.post(URL, json=payload, headers=token).status_code == 422


def test_more_than_5000_books_is_422(api: TestClient, token: dict[str, str]) -> None:
    payload = {"books": [tiny_book(i) for i in range(5001)], "interactions": []}

    response = api.post(URL, json=payload, headers=token)

    assert response.status_code == 422


def test_no_books_is_422(api: TestClient, token: dict[str, str]) -> None:
    assert api.post(URL, json={"books": [], "interactions": []}, headers=token).status_code == 422


def test_the_caps_on_books_interactions_and_limit_are_exact() -> None:
    def request(books: int = 1, interactions: int = 0, limit: int = 10) -> BatchRequest:
        return BatchRequest.model_validate(
            {
                "books": [tiny_book(i) for i in range(books)],
                "interactions": [{"user_id": "u", "book_id": "b0", "weight": 1}] * interactions,
                "limit": limit,
            }
        )

    request(books=5000)
    request(interactions=200_000)
    request(limit=1)
    request(limit=50)
    for bad in (
        {"books": 5001},
        {"books": 0},
        {"interactions": 200_001},
        {"limit": 0},
        {"limit": 51},
    ):
        with pytest.raises(ValueError):
            request(**bad)


def test_the_old_per_user_stub_route_is_gone(api: TestClient, token: dict[str, str]) -> None:
    response = api.post("/recommend/books", json={"user_id": "u1", "limit": 10}, headers=token)

    assert response.status_code == 404


# --- the committed contract fixture, shared byte for byte with the backend ---


def test_the_committed_contract_fixture_still_matches(api: TestClient, token: dict[str, str]) -> None:
    raw = FIXTURE.read_bytes()
    assert b"\r" not in raw  # LF only: the two repos' copies must stay byte-identical
    fixture = json.loads(raw)
    assert set(fixture) == {"request", "response"}
    assert len(fixture["request"]["interactions"]) == 12
    assert len({i["user_id"] for i in fixture["request"]["interactions"]}) == 4

    response = api.post(URL, json=fixture["request"], headers=token)

    assert response.status_code == 200
    got, want = response.json(), fixture["response"]
    assert got["model_version"] == want["model_version"]
    assert [user["user_id"] for user in got["users"]] == [user["user_id"] for user in want["users"]]
    for got_user, want_user in zip(got["users"], want["users"], strict=True):
        assert [r["book_id"] for r in got_user["results"]] == [r["book_id"] for r in want_user["results"]]
        assert [r["method"] for r in got_user["results"]] == [r["method"] for r in want_user["results"]]
        assert [r["score"] for r in got_user["results"]] == pytest.approx(
            [r["score"] for r in want_user["results"]]
        )
