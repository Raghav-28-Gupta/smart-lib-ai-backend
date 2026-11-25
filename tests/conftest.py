"""Shared fixtures. Nothing here touches the network.

The `client` fixtures never see the committed model unless a test asks for it (`committed_client`).
`trained_small` trains a tiny model once per session for the tests that need a real one.
"""

from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.noshow import get_model, load_committed_model

# The date a test artifact is named after: derived from today, never a fixed calendar date.
TEST_TRAIN_DATE = date.today() - timedelta(days=3)
SMALL_STUDENTS = 300
SMALL_SEED = 42


class FakeModel:
    """Stands in for the trained model: same predict() shape as the NoShowModel Protocol."""

    def __init__(
        self,
        probability: float = 0.31,
        penalty_factor: float = 0.155,
        model_version: str = "noshow-test",
    ) -> None:
        self.probability = probability
        self.penalty_factor = penalty_factor
        self.model_version = model_version
        self.calls: list[dict[str, Any]] = []

    def predict(self, features: dict[str, Any]) -> tuple[float, float, str]:
        self.calls.append(dict(features))
        return self.probability, self.penalty_factor, self.model_version


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """A fresh, valid request body (a room booking by a fairly reliable student)."""
    return {
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
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    """The real app with `get_model` overridden to answer None, whatever artifact is committed.

    The 503 tests are about the contract (no model means 503 `model_unavailable`), so they must not
    depend on whether a trained artifact exists. The tests of the real artifact use `committed_client`
    or build their own app state.
    """
    app.dependency_overrides[get_model] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def client_with_model(fake_model: FakeModel) -> Iterator[TestClient]:
    """The app with get_model overridden to answer with the fake model."""
    app.dependency_overrides[get_model] = lambda: fake_model
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def fresh_model_cache() -> Iterator[None]:
    """Forget the once-loaded committed model before and after the test.

    For tests that point the loader at another directory or count how often it loads.
    """
    load_committed_model.cache_clear()
    yield
    load_committed_model.cache_clear()


@pytest.fixture
def committed_client(fresh_model_cache: None) -> Iterator[TestClient]:
    """The real app, nothing overridden: it serves the artifact committed under app/ml/artifacts."""
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def trained_small(tmp_path_factory: pytest.TempPathFactory):
    """A real model trained on a few hundred simulated students, in a temporary directory.

    Session scoped: training takes a few seconds, and the tests only read the result.
    """
    from app.ml.noshow.train import train

    out_dir = tmp_path_factory.mktemp("trained_small")
    return train(seed=SMALL_SEED, students=SMALL_STUDENTS, out_dir=out_dir, train_date=TEST_TRAIN_DATE)
