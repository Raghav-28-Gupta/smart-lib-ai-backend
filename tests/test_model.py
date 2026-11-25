"""Serving the trained no-show model: the bounded penalty, the loader, the endpoint, and the
committed artifact (spec 2026-11-24-no-show-handling, sections 6 and 8).

The model-quality targets live at the bottom, against the artifact that is committed in the repo.
"""

import hashlib
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from app.ml.noshow import features as features_module
from app.ml.noshow import model as model_module
from app.ml.noshow import train as train_module
from app.ml.noshow.features import (
    FEATURE_ORDER,
    MAX_PENALTY_CAP,
    RESOURCE_TYPE_CODES,
    encode_features,
)
from app.ml.noshow.model import (
    ARTIFACTS_DIR,
    LogOddsClassifier,
    NoArtifactError,
    NoShowModel,
    find_newest_artifact,
    penalty_factor,
)
from app.ml.noshow.simulate import simulate
from tests.conftest import SMALL_SEED, SMALL_STUDENTS

URL = "/predict/no-show"
VERSION_SHAPE = re.compile(r"^noshow-\d{8}-[0-9a-f]{8}$")


# --- the bounded penalty ------------------------------------------------------------------------


def test_penalty_factor_is_within_0_and_the_cap_for_every_probability_including_the_ends():
    probabilities = [index / 1000 for index in range(1001)]  # 0.0 ... 1.0 inclusive

    penalties = [penalty_factor(p) for p in probabilities]

    assert min(penalties) == 0.0
    assert max(penalties) == MAX_PENALTY_CAP == 0.2
    assert all(0.0 <= penalty <= 0.2 for penalty in penalties)


def test_penalty_factor_is_monotone_in_the_probability():
    probabilities = [index / 1000 for index in range(1001)]

    penalties = [penalty_factor(p) for p in probabilities]

    assert all(later >= earlier for earlier, later in zip(penalties, penalties[1:]))


def test_penalty_factor_is_half_the_probability_until_the_cap():
    assert penalty_factor(0.0) == 0.0
    assert penalty_factor(0.1) == pytest.approx(0.05)
    assert penalty_factor(0.3) == pytest.approx(0.15)
    assert penalty_factor(0.4) == pytest.approx(0.2)
    assert penalty_factor(0.9) == 0.2
    assert penalty_factor(1.0) == 0.2


def test_penalty_factor_clamps_a_probability_outside_0_1():
    assert penalty_factor(-0.3) == 0.0
    assert penalty_factor(7.0) == 0.2


def test_penalty_factor_takes_its_alpha_and_cap_as_arguments():
    assert penalty_factor(0.1, alpha=1.0) == pytest.approx(0.1)
    assert penalty_factor(0.5, alpha=1.0, cap=0.3) == 0.3
    assert penalty_factor(0.5, alpha=0.0) == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_penalty_factor_rejects_a_probability_that_is_not_finite(bad):
    with pytest.raises(ValueError):
        penalty_factor(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_penalty_factor_rejects_a_non_finite_alpha_or_cap(bad):
    with pytest.raises(ValueError):
        penalty_factor(0.2, alpha=bad)
    with pytest.raises(ValueError):
        penalty_factor(0.2, cap=bad)


# --- one encoder for training and serving -------------------------------------------------------


@pytest.fixture(scope="module")
def frame_small():
    return simulate(SMALL_STUDENTS, SMALL_SEED)


ROW = {
    "hour_of_day": 19,
    "day_of_week": 4,
    "duration_minutes": 90,
    "lead_time_hours": 30.5,
    "resource_type": "room",
    "past_attendance_ratio": 0.8,
    "consecutive_attended_count": 3,
    "recent_noshows_count": 1,
    "reliability_score": 82.0,
    "bookings_last_30d": 6,
}


def test_encode_features_makes_one_float_row_in_feature_order_with_seat_0_and_room_1():
    seat = encode_features({**ROW, "resource_type": "seat"})
    room = encode_features(ROW)

    assert list(room.columns) == FEATURE_ORDER
    assert room.shape == (1, 10)
    assert set(room.dtypes.astype(str)) == {"float64"}
    assert seat.loc[0, "resource_type"] == 0.0
    assert room.loc[0, "resource_type"] == 1.0
    assert room.loc[0, "lead_time_hours"] == 30.5


def test_encode_features_ignores_the_order_of_the_keys_and_extra_keys():
    shuffled = dict(reversed(list(ROW.items())))
    shuffled["something_else"] = 99

    assert encode_features(shuffled).equals(encode_features(ROW))


def test_encode_features_encodes_a_frame_the_same_way_as_a_row(frame_small):
    encoded = encode_features(frame_small)

    assert list(encoded.columns) == FEATURE_ORDER
    assert len(encoded) == len(frame_small)
    for position in (0, 1, 17, len(frame_small) // 2, len(frame_small) - 1):
        row = frame_small.iloc[position][FEATURE_ORDER].to_dict()
        assert encode_features(row).iloc[0].to_dict() == encoded.iloc[position].to_dict()


def test_encode_features_refuses_an_unknown_resource_type():
    with pytest.raises(ValueError, match="resource_type"):
        encode_features({**ROW, "resource_type": "desk"})


def test_encode_features_refuses_a_missing_feature():
    row = dict(ROW)
    del row["bookings_last_30d"]

    with pytest.raises(ValueError, match="bookings_last_30d"):
        encode_features(row)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_encode_features_refuses_a_value_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        encode_features({**ROW, "lead_time_hours": bad})


def test_encode_features_refuses_an_integer_too_large_for_a_float():
    """A 400-digit integer cannot become a float64: a clear ValueError, not an OverflowError."""
    with pytest.raises(ValueError):
        encode_features({**ROW, "bookings_last_30d": 10**400})


def test_training_and_serving_share_one_encoder():
    """They cannot drift because there is only one function, imported by both."""
    assert train_module.encode_features is features_module.encode_features
    assert model_module.encode_features is features_module.encode_features


def test_a_served_row_scores_exactly_what_the_training_path_scores_for_it(trained_small, frame_small):
    """The training path encodes a whole frame; serving encodes a dict. Same row, same probability."""
    model = NoShowModel.load(trained_small.artifact_path.parent)
    batch = model.calibrated.predict_proba(encode_features(frame_small))[:, 1]

    for position in range(0, len(frame_small), max(1, len(frame_small) // 60)):
        row = frame_small.iloc[position][FEATURE_ORDER].to_dict()
        probability, _, _ = model.predict(row)
        assert probability == pytest.approx(float(batch[position]), abs=1e-12)


# --- the log-odds wrapper the calibrator is fitted through --------------------------------------


@pytest.fixture(scope="module")
def fitted_xgboost(frame_small):
    from xgboost import XGBClassifier

    features = encode_features(frame_small)
    return XGBClassifier(n_estimators=30, max_depth=3, random_state=0, tree_method="hist", n_jobs=1).fit(
        features, frame_small["no_show"].to_numpy()
    )


def test_the_wrapper_offers_log_odds_so_scikit_learn_calibrates_on_them(fitted_xgboost, frame_small):
    """XGBClassifier has no decision_function, so without this scikit-learn fits a sigmoid to the
    probability itself, which cannot reach below about 0.06 whatever the model says."""
    wrapper = LogOddsClassifier(fitted_xgboost)
    features = encode_features(frame_small)
    probability = fitted_xgboost.predict_proba(features)[:, 1]

    assert not hasattr(fitted_xgboost, "decision_function")
    assert wrapper.decision_function(features) == pytest.approx(np.log(probability / (1 - probability)), abs=1e-3)


def test_the_wrapper_answers_like_the_model_it_holds(fitted_xgboost, frame_small):
    wrapper = LogOddsClassifier(fitted_xgboost)
    features = encode_features(frame_small)

    assert list(wrapper.classes_) == [0, 1]
    assert np.array_equal(wrapper.predict_proba(features), fitted_xgboost.predict_proba(features))
    assert np.array_equal(wrapper.predict(features), fitted_xgboost.predict(features))


def _dumped(obj: Any) -> io.BytesIO:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return buffer


def test_the_wrapper_lives_in_an_app_module_so_the_artifact_can_unpickle_it(fitted_xgboost, frame_small):
    assert LogOddsClassifier.__module__ == "app.ml.noshow.model"
    features = encode_features(frame_small)
    original = LogOddsClassifier(fitted_xgboost)

    again = joblib.load(_dumped(original))

    assert np.array_equal(again.decision_function(features), original.decision_function(features))


# --- choosing the newest artifact ---------------------------------------------------------------


def _touch(directory: Path, name: str, mtime: int | None = None) -> Path:
    path = directory / name
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_the_newest_artifact_is_the_latest_date_in_its_name_whatever_the_file_times(tmp_path):
    """A git checkout gives every file the same time, so the date in the name decides."""
    _touch(tmp_path, "noshow-20260105-aaaaaaaa.joblib", mtime=3_000_000_000)
    newest = _touch(tmp_path, "noshow-20261124-bbbbbbbb.joblib", mtime=1_000_000_000)
    _touch(tmp_path, "noshow-20260615-cccccccc.joblib", mtime=2_000_000_000)

    assert find_newest_artifact(tmp_path) == newest


def test_artifacts_from_one_day_are_ordered_by_file_time_then_by_name(tmp_path):
    _touch(tmp_path, "noshow-20261124-11111111.joblib", mtime=1_000_000_000)
    newer = _touch(tmp_path, "noshow-20261124-22222222.joblib", mtime=2_000_000_000)
    _touch(tmp_path, "noshow-20261124-33333333.joblib", mtime=1_500_000_000)
    assert find_newest_artifact(tmp_path) == newer

    # Equal file times: the higher name wins, so the choice never depends on directory order.
    for path in tmp_path.iterdir():
        os.utime(path, (2_000_000_000, 2_000_000_000))
    assert find_newest_artifact(tmp_path).name == "noshow-20261124-33333333.joblib"


def test_files_that_are_not_artifacts_are_never_chosen(tmp_path):
    for name in (
        "metrics.json",
        "noshow-latest.joblib",
        "noshow-20261124-ZZZZZZZZ.joblib",  # hash is not hex
        "noshow-20261124-abc.joblib",  # hash too short
        "noshow-20261340-abcdef01.joblib",  # not a date
        "noshow-20261124-abcdef01.joblib.bak",
        "other-20261124-abcdef01.joblib",
    ):
        _touch(tmp_path, name)
    (tmp_path / "noshow-20261201-abcdef01.joblib").mkdir()  # a directory with an artifact's name

    assert find_newest_artifact(tmp_path) is None


def test_no_artifact_when_the_directory_is_empty_or_missing(tmp_path):
    assert find_newest_artifact(tmp_path) is None
    assert find_newest_artifact(tmp_path / "missing") is None


def test_loading_from_a_directory_without_an_artifact_says_so(tmp_path):
    with pytest.raises(NoArtifactError):
        NoShowModel.load(tmp_path)


def test_load_picks_the_newest_artifact_by_name_date(tmp_path, trained_small):
    """Two real artifacts: the one named for the later day is served, not the one touched last."""
    source = trained_small.artifact_path
    data = source.read_bytes()
    older_name = "noshow-20250101-" + hashlib.sha256(data).hexdigest()[:8] + ".joblib"
    (tmp_path / older_name).write_bytes(data)
    os.utime(tmp_path / older_name, (3_000_000_000, 3_000_000_000))  # touched last, dated first
    (tmp_path / source.name).write_bytes(data)
    os.utime(tmp_path / source.name, (1_000_000_000, 1_000_000_000))

    model = NoShowModel.load(tmp_path)

    assert model.version == source.stem


# --- loading and predicting ---------------------------------------------------------------------


def test_the_version_is_the_artifact_stem_and_has_the_documented_shape(trained_small):
    model = NoShowModel.load(trained_small.artifact_path.parent)

    assert model.version == trained_small.artifact_path.stem
    assert VERSION_SHAPE.match(model.version)


def test_predict_returns_a_probability_a_bounded_penalty_and_the_version(trained_small):
    model = NoShowModel.load(trained_small.artifact_path.parent)

    probability, penalty, version = model.predict(dict(ROW))

    assert 0.0 <= probability <= 1.0
    assert penalty == penalty_factor(probability)
    assert 0.0 <= penalty <= MAX_PENALTY_CAP
    assert version == trained_small.artifact_path.stem


class _Constant:
    """Stands in for a calibrated model that answers one number for every row."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, frame):
        return np.column_stack([np.full(len(frame), 1 - self.value), np.full(len(frame), self.value)])


@pytest.mark.parametrize(
    "answer, expected",
    [(1.2, 1.0), (-0.3, 0.0), (0.0, 0.0), (1.0, 1.0), (0.25, 0.25)],
)
def test_predict_clamps_the_probability_into_0_1(answer, expected):
    model = NoShowModel(_Constant(answer), "noshow-20260101-abcdef01")

    probability, penalty, _ = model.predict(dict(ROW))

    assert probability == expected
    assert 0.0 <= penalty <= MAX_PENALTY_CAP


def test_predict_refuses_a_model_answer_that_is_not_a_number():
    model = NoShowModel(_Constant(math.nan), "noshow-20260101-abcdef01")

    with pytest.raises(ValueError):
        model.predict(dict(ROW))


def test_predict_refuses_an_unknown_resource_type(trained_small):
    model = NoShowModel.load(trained_small.artifact_path.parent)

    with pytest.raises(ValueError):
        model.predict({**ROW, "resource_type": "desk"})


def _mean_probability_by_recent_noshows(model: NoShowModel, fresh) -> list[float]:
    """Mean predicted probability of fresh simulated bookings with 0, 1 and 2 or more recent no-shows.

    Observed groups, not a counterfactual: setting `recent_noshows_count` alone while holding the
    reliability score and attendance ratio fixed describes a student who cannot exist (those three
    move together), and a tree model is free to answer such a row any way it likes.
    """
    scores = model.calibrated.predict_proba(encode_features(fresh))[:, 1]
    recent = fresh["recent_noshows_count"].to_numpy()
    return [float(scores[mask].mean()) for mask in (recent == 0, recent == 1, recent >= 2)]


def test_probability_rises_on_average_with_recent_noshows_count(trained_small):
    """On students the model never saw, the more recent no-shows, the higher the mean probability."""
    model = NoShowModel.load(trained_small.artifact_path.parent)

    none, one, several = _mean_probability_by_recent_noshows(model, simulate(300, 9001))

    assert none < one < several


# --- a bad artifact never loads -----------------------------------------------------------------


def _write_artifact(directory: Path, payload_bytes: bytes, *, honest_hash: bool = True) -> Path:
    digest = hashlib.sha256(payload_bytes).hexdigest()[:8] if honest_hash else "deadbeef"
    path = directory / f"noshow-20260101-{digest}.joblib"
    path.write_bytes(payload_bytes)
    return path


def _dump(payload: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    return buffer.getvalue()


def test_an_artifact_whose_bytes_do_not_match_its_hash_is_refused(tmp_path, trained_small):
    data = trained_small.artifact_path.read_bytes()
    path = _write_artifact(tmp_path, data[: len(data) // 2], honest_hash=False)
    assert path.exists()

    with pytest.raises(model_module.ArtifactError, match="hash"):
        NoShowModel.load(tmp_path)


def test_an_artifact_that_is_not_a_pickle_is_refused(tmp_path):
    _write_artifact(tmp_path, b"this is not a model")

    with pytest.raises(model_module.ArtifactError):
        NoShowModel.load(tmp_path)


def test_an_artifact_for_another_feature_order_is_refused(tmp_path):
    payload = {"model": _Constant(0.1), "feature_order": list(reversed(FEATURE_ORDER)), "encoding": {}}
    _write_artifact(tmp_path, _dump(payload))

    with pytest.raises(model_module.ArtifactError, match="feature"):
        NoShowModel.load(tmp_path)


def test_an_artifact_with_another_encoding_is_refused(tmp_path):
    payload = {
        "model": _Constant(0.1),
        "feature_order": FEATURE_ORDER,
        "encoding": {"resource_type": {"seat": 1, "room": 0}},
    }
    _write_artifact(tmp_path, _dump(payload))

    with pytest.raises(model_module.ArtifactError, match="encoding"):
        NoShowModel.load(tmp_path)


def test_an_artifact_that_is_not_a_dict_is_refused(tmp_path):
    _write_artifact(tmp_path, _dump([1, 2, 3]))

    with pytest.raises(model_module.ArtifactError):
        NoShowModel.load(tmp_path)


class _Broken:
    """Unpickles fine, then fails the moment it is asked for a probability."""

    def predict_proba(self, frame):
        raise RuntimeError("boom: the secret text of the error")


class _WrongShape:
    def predict_proba(self, frame):
        return np.zeros((len(frame), 1))


def _good_payload(model: Any) -> dict[str, Any]:
    return {"model": model, "feature_order": FEATURE_ORDER, "encoding": {"resource_type": RESOURCE_TYPE_CODES}}


@pytest.mark.parametrize("model", [_Broken(), _WrongShape(), _Constant(math.nan)])
def test_a_model_that_unpickles_but_cannot_predict_is_refused_at_load(tmp_path, model):
    """One test prediction at load time: a broken model is a 503 from startup, not a 500 per request."""
    _write_artifact(tmp_path, _dump(_good_payload(model)))

    with pytest.raises(model_module.ArtifactError, match="test prediction"):
        NoShowModel.load(tmp_path)


def test_a_model_that_can_predict_passes_the_load_time_check(tmp_path):
    _write_artifact(tmp_path, _dump(_good_payload(_Constant(0.25))))

    assert NoShowModel.load(tmp_path).predict(dict(ROW))[0] == 0.25


def test_a_broken_model_makes_the_service_answer_503_from_startup(
    tmp_path, monkeypatch, fresh_model_cache, caplog, valid_payload
):
    from fastapi.testclient import TestClient

    from app.main import app

    _write_artifact(tmp_path, _dump(_good_payload(_Broken())))
    monkeypatch.setattr(model_module, "ARTIFACTS_DIR", tmp_path)
    app.dependency_overrides.clear()

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            response = client.post(URL, json=valid_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("RuntimeError" in message for message in warnings)  # the class of the error only
    assert not any("secret text" in message for message in warnings)


# --- the endpoint with a real model -------------------------------------------------------------


def test_the_service_answers_503_not_a_crash_for_each_kind_of_bad_artifact(
    tmp_path, monkeypatch, fresh_model_cache, caplog, valid_payload, trained_small
):
    from fastapi.testclient import TestClient

    from app.main import app

    data = trained_small.artifact_path.read_bytes()
    _write_artifact(tmp_path, data[:100], honest_hash=False)  # a truncated copy: its bytes do not match its name
    monkeypatch.setattr(model_module, "ARTIFACTS_DIR", tmp_path)
    app.dependency_overrides.clear()

    with caplog.at_level(logging.WARNING):
        with TestClient(app) as client:
            response = client.post(URL, json=valid_payload)
            health = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}
    assert health.status_code == 200
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("no-show model" in message for message in warnings)
    # The class of the error is logged, never the path or the exception text.
    assert not any(str(tmp_path) in message for message in warnings)


def test_the_service_answers_503_when_the_directory_has_no_artifact(
    tmp_path, monkeypatch, fresh_model_cache, valid_payload
):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(model_module, "ARTIFACTS_DIR", tmp_path)
    app.dependency_overrides.clear()

    with TestClient(app) as client:
        response = client.post(URL, json=valid_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}


def test_a_good_artifact_in_another_directory_is_served(
    tmp_path, monkeypatch, fresh_model_cache, valid_payload, trained_small
):
    from fastapi.testclient import TestClient

    from app.main import app

    (tmp_path / trained_small.artifact_path.name).write_bytes(trained_small.artifact_path.read_bytes())
    monkeypatch.setattr(model_module, "ARTIFACTS_DIR", tmp_path)
    app.dependency_overrides.clear()

    with TestClient(app) as client:
        response = client.post(URL, json=valid_payload)

    assert response.status_code == 200
    assert response.json()["model_version"] == trained_small.artifact_path.stem


def test_the_artifact_is_loaded_once_not_on_every_request(
    tmp_path, monkeypatch, fresh_model_cache, valid_payload, trained_small
):
    from fastapi.testclient import TestClient

    from app.main import app

    (tmp_path / trained_small.artifact_path.name).write_bytes(trained_small.artifact_path.read_bytes())
    monkeypatch.setattr(model_module, "ARTIFACTS_DIR", tmp_path)
    app.dependency_overrides.clear()
    loads = []
    real_load = NoShowModel.load.__func__
    monkeypatch.setattr(
        NoShowModel, "load", classmethod(lambda cls, *args, **kwargs: loads.append(1) or real_load(cls, *args, **kwargs))
    )

    with TestClient(app) as client:
        for _ in range(3):
            assert client.post(URL, json=valid_payload).status_code == 200

    assert len(loads) == 1


# --- the committed artifact ---------------------------------------------------------------------


def _committed_metrics() -> dict[str, Any]:
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))


def test_a_trained_artifact_and_metrics_are_committed():
    artifact = find_newest_artifact(ARTIFACTS_DIR)

    assert artifact is not None, "run `python -m app.ml.noshow.train --seed 42` and commit the result"
    assert (ARTIFACTS_DIR / "metrics.json").is_file()


def test_model_version_has_the_documented_shape_and_names_the_committed_file():
    metrics = _committed_metrics()
    artifact = find_newest_artifact(ARTIFACTS_DIR)

    assert VERSION_SHAPE.match(metrics["model_version"])
    assert metrics["artifact"] == artifact.name
    assert metrics["model_version"] == artifact.stem
    assert artifact.stem.rsplit("-", 1)[1] == hashlib.sha256(artifact.read_bytes()).hexdigest()[:8]


def test_the_endpoint_serves_the_committed_artifact(committed_client, valid_payload):
    response = committed_client.post(URL, json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == _committed_metrics()["model_version"]
    assert 0.0 <= body["predicted_probability"] <= 1.0
    assert body["penalty_factor"] == pytest.approx(min(body["predicted_probability"] * 0.5, 0.2))


def test_the_committed_model_scores_a_risky_booking_above_a_safe_one(committed_client, valid_payload):
    safe = {
        **valid_payload,
        "lead_time_hours": 2.0,
        "hour_of_day": 11,
        "resource_type": "seat",
        "past_attendance_ratio": 1.0,
        "recent_noshows_count": 0,
        "reliability_score": 100,
    }
    risky = {
        **valid_payload,
        "lead_time_hours": 200.0,
        "hour_of_day": 21,
        "resource_type": "room",
        "past_attendance_ratio": 0.4,
        "recent_noshows_count": 3,
        "reliability_score": 40,
    }

    safe_body = committed_client.post(URL, json=safe).json()
    risky_body = committed_client.post(URL, json=risky).json()

    assert risky_body["predicted_probability"] > safe_body["predicted_probability"] + 0.15
    assert risky_body["penalty_factor"] > safe_body["penalty_factor"]


@pytest.fixture(scope="module")
def holdout():
    """The committed model scored on students it has never seen: another seed, another world."""
    model = NoShowModel.load(ARTIFACTS_DIR)
    fresh = simulate(500, 777)
    labels = fresh["no_show"].to_numpy()
    scores = model.calibrated.predict_proba(encode_features(fresh))[:, 1]
    return labels, scores


def test_holdout_roc_auc_is_at_least_0_70(holdout):
    from sklearn.metrics import roc_auc_score

    labels, scores = holdout

    assert roc_auc_score(labels, scores) >= 0.70
    assert _committed_metrics()["test"]["roc_auc"] >= 0.70


def test_holdout_brier_score_beats_the_base_rate_predictor(holdout):
    from sklearn.metrics import brier_score_loss

    labels, scores = holdout
    train_mean = _committed_metrics()["base_rate"]["train_mean"]
    base = float(np.mean((train_mean - labels) ** 2))

    assert brier_score_loss(labels, scores) < base
    committed = _committed_metrics()["test"]
    assert committed["brier"] < committed["base_rate_brier"]


def test_calibration_the_mean_absolute_gap_across_bins_is_at_most_0_05(holdout):
    labels, scores = holdout
    bins = [entry for entry in train_module.calibration_bins(labels, scores) if entry["count"] > 0]
    gap = sum(abs(entry["mean_predicted"] - entry["observed_rate"]) for entry in bins) / len(bins)

    assert gap <= 0.05
    assert _committed_metrics()["test"]["mean_abs_calibration_gap"] <= 0.05


def test_the_committed_model_never_says_never_or_certainly(holdout):
    """Isotonic's flat ends gave exactly 0.0 and 1.0 for bookings that were nothing of the kind."""
    _, scores = holdout

    assert scores.min() > 0.0
    assert scores.max() < 1.0


def _raw_probabilities(model: NoShowModel, frame) -> np.ndarray:
    """What the classifier inside the calibrator says before calibration."""
    raw_model = model.calibrated.calibrated_classifiers_[0].estimator.estimator
    return raw_model.predict_proba(encode_features(frame))[:, 1]


@pytest.fixture(scope="module")
def holdout_with_raw():
    model = NoShowModel.load(ARTIFACTS_DIR)
    fresh = simulate(500, 777)
    served = model.calibrated.predict_proba(encode_features(fresh))[:, 1]
    return fresh["no_show"].to_numpy(), served, _raw_probabilities(model, fresh)


def test_calibrating_does_not_make_the_committed_model_worse_than_the_raw_one(holdout_with_raw):
    from sklearn.metrics import brier_score_loss

    labels, served, raw = holdout_with_raw
    committed = _committed_metrics()

    assert brier_score_loss(labels, served) <= brier_score_loss(labels, raw) + 0.001
    assert committed["test"]["brier"] <= committed["uncalibrated_test"]["brier"] + 0.001


def test_the_lowest_decile_is_predicted_within_0_02_of_what_is_observed(holdout):
    """Fitted to the probability instead of the log-odds, the calibrator served 0.074 for a decile
    whose students missed 0.040 of their bookings."""
    labels, scores = holdout

    lowest = train_module.calibration_bins(labels, scores)[0]
    assert abs(lowest["mean_predicted"] - lowest["observed_rate"]) <= 0.02
    committed = _committed_metrics()["calibration_bins"][0]
    assert abs(committed["mean_predicted"] - committed["observed_rate"]) <= 0.02


def test_the_calibrator_leaves_the_raw_models_low_scores_nearly_unchanged(holdout_with_raw):
    """A sigmoid on the probability cannot output below 0.0626 and moved 26 percent of bookings up."""
    _, served, raw = holdout_with_raw

    assert raw.min() < 0.03
    assert served.min() < 0.04
    assert np.abs(served - raw).max() < 0.05
    assert _committed_metrics()["test"]["min_probability"] < 0.04


def test_probability_rises_on_average_with_recent_noshows_in_the_committed_model():
    model = NoShowModel.load(ARTIFACTS_DIR)

    none, one, several = _mean_probability_by_recent_noshows(model, simulate(500, 778))

    assert none < one < several


def test_the_committed_metrics_record_the_seed_and_the_documented_parameters():
    metrics = _committed_metrics()

    assert metrics["seed"] == 42
    assert metrics["students"]["total"] == 2000
    assert metrics["xgboost"]["n_estimators"] == 150
    assert metrics["xgboost"]["max_depth"] == 4
    assert metrics["xgboost"]["learning_rate"] == 0.05
    assert metrics["calibration"]["method"] in {"isotonic", "sigmoid"}
    assert len(metrics["calibration_bins"]) == 10
    assert "trained_at" not in json.dumps(metrics)
