"""The training pipeline (spec 2026-11-24-no-show-handling, section 8): split by student, fit,
calibrate, evaluate, write the artifact and metrics.json.

Everything here trains on a few hundred simulated students (seconds, not minutes). The model
quality targets (AUC, Brier, calibration) are checked against the committed artifact in
test_model.py, where the full-size model is.
"""

import hashlib
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.ml.noshow import train as train_module
from app.ml.noshow.features import FEATURE_ORDER
from app.ml.noshow.simulate import simulate
from app.ml.noshow.model import LogOddsClassifier
from app.ml.noshow.train import calibrate, calibration_bins, choose_calibration, main, split_by_student, train
from tests.conftest import SMALL_SEED, SMALL_STUDENTS, TEST_TRAIN_DATE

ARTIFACT_NAME = re.compile(r"^noshow-(\d{8})-([0-9a-f]{8})\.joblib$")


@pytest.fixture(scope="module")
def frame():
    return simulate(SMALL_STUDENTS, SMALL_SEED)


# --- the split ----------------------------------------------------------------------------------


def test_the_split_is_by_student_with_no_student_in_two_sets(frame):
    fit, calibrate, test = split_by_student(frame, SMALL_SEED)

    fit_students = set(fit["student_id"])
    calibrate_students = set(calibrate["student_id"])
    test_students = set(test["student_id"])

    assert fit_students.isdisjoint(calibrate_students)
    assert fit_students.isdisjoint(test_students)
    assert calibrate_students.isdisjoint(test_students)
    # Every student lands somewhere, and every row goes with its student.
    assert fit_students | calibrate_students | test_students == set(frame["student_id"])
    assert len(fit) + len(calibrate) + len(test) == len(frame)


def test_the_split_is_about_60_20_20_students(frame):
    fit, calibrate, test = split_by_student(frame, SMALL_SEED)

    counts = [fit["student_id"].nunique(), calibrate["student_id"].nunique(), test["student_id"].nunique()]
    assert counts == [180, 60, 60]


def test_the_split_is_deterministic_for_a_seed_and_differs_for_another(frame):
    first = split_by_student(frame, SMALL_SEED)
    again = split_by_student(frame, SMALL_SEED)
    other = split_by_student(frame, SMALL_SEED + 1)

    for left, right in zip(first, again):
        assert left.index.equals(right.index)
    assert not first[0].index.equals(other[0].index)


def test_the_split_refuses_too_few_students():
    tiny = simulate(4, 1)

    with pytest.raises(ValueError, match="students"):
        split_by_student(tiny, 1)


def test_training_stops_if_a_student_would_be_in_two_splits(monkeypatch, frame, tmp_path):
    fit, calibrate, test = split_by_student(frame, SMALL_SEED)
    leaky_calibrate = calibrate.copy()
    leaky_calibrate.loc[leaky_calibrate.index[0], "student_id"] = fit["student_id"].iloc[0]
    monkeypatch.setattr(train_module, "split_by_student", lambda *_: (fit, leaky_calibrate, test))

    with pytest.raises(RuntimeError, match="student"):
        train(seed=SMALL_SEED, students=SMALL_STUDENTS, out_dir=tmp_path, train_date=TEST_TRAIN_DATE)

    assert list(tmp_path.iterdir()) == []


# --- reproducibility ----------------------------------------------------------------------------


def test_training_twice_with_one_seed_gives_identical_metrics(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = train(seed=7, students=200, out_dir=first_dir, train_date=TEST_TRAIN_DATE)
    second = train(seed=7, students=200, out_dir=second_dir, train_date=TEST_TRAIN_DATE)

    assert first.metrics == second.metrics
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    # The artifact itself is the same bytes, so the version (which hashes them) is the same too.
    assert first.artifact_path.name == second.artifact_path.name
    assert first.artifact_path.read_bytes() == second.artifact_path.read_bytes()


def test_a_different_seed_gives_different_metrics(tmp_path):
    first = train(seed=7, students=200, out_dir=tmp_path / "a", train_date=TEST_TRAIN_DATE)
    other = train(seed=8, students=200, out_dir=tmp_path / "b", train_date=TEST_TRAIN_DATE)

    assert first.metrics["test"] != other.metrics["test"]
    assert first.model_version != other.model_version


# --- what metrics.json holds --------------------------------------------------------------------


def test_metrics_json_is_the_returned_metrics(trained_small):
    on_disk = json.loads(trained_small.metrics_path.read_text(encoding="utf-8"))

    assert on_disk == trained_small.metrics
    assert trained_small.metrics_path.name == "metrics.json"
    assert trained_small.metrics_path.parent == trained_small.artifact_path.parent


def test_metrics_json_has_the_documented_content(trained_small):
    metrics = trained_small.metrics

    assert metrics["seed"] == SMALL_SEED
    assert metrics["students"]["total"] == SMALL_STUDENTS
    for key in (
        "roc_auc",
        "brier",
        "log_loss",
        "base_rate_brier",
        "mean_abs_calibration_gap",
        "min_probability",
        "max_probability",
    ):
        assert isinstance(metrics["test"][key], float), key
    assert set(metrics["feature_importances"]) == set(FEATURE_ORDER)
    for library in ("python", "xgboost", "scikit_learn", "numpy", "pandas"):
        assert metrics["libraries"][library], library


def test_metrics_json_records_no_timestamp(trained_small):
    """Reproducibility: the same seed must give the same file, so nothing records when it ran."""
    text = trained_small.metrics_path.read_text(encoding="utf-8").lower()

    assert "trained_at" not in text
    assert "timestamp" not in text


def test_row_and_student_counts_per_split_add_up(trained_small, frame):
    metrics = trained_small.metrics

    assert metrics["rows"]["total"] == len(frame)
    assert metrics["rows"]["train"] + metrics["rows"]["calibration"] + metrics["rows"]["test"] == len(frame)
    students = metrics["students"]
    assert students["train"] + students["calibration"] + students["test"] == students["total"]
    assert min(metrics["rows"][name] for name in ("train", "calibration", "test")) > 0


def test_the_base_rate_brier_is_the_train_mean_predicted_for_every_test_row(trained_small, frame):
    fit, _, test = split_by_student(frame, SMALL_SEED)
    train_mean = float(fit["no_show"].mean())
    expected = float(np.mean((train_mean - test["no_show"].to_numpy()) ** 2))

    assert trained_small.metrics["test"]["base_rate_brier"] == pytest.approx(expected, abs=1e-6)
    assert trained_small.metrics["base_rate"]["train_mean"] == pytest.approx(train_mean, abs=1e-6)


def test_ten_calibration_bins_cover_every_test_row(trained_small):
    bins = trained_small.metrics["calibration_bins"]

    assert len(bins) == 10
    for entry in bins:
        assert set(entry) == {"mean_predicted", "observed_rate", "count"}
        assert 0.0 <= entry["mean_predicted"] <= 1.0
        assert 0.0 <= entry["observed_rate"] <= 1.0
    assert sum(entry["count"] for entry in bins) == trained_small.metrics["rows"]["test"]


def test_the_calibration_gap_is_the_mean_absolute_gap_over_populated_bins(trained_small):
    bins = [entry for entry in trained_small.metrics["calibration_bins"] if entry["count"] > 0]
    expected = sum(abs(entry["mean_predicted"] - entry["observed_rate"]) for entry in bins) / len(bins)

    assert trained_small.metrics["test"]["mean_abs_calibration_gap"] == pytest.approx(expected, abs=2e-6)


def test_calibration_bins_by_hand():
    """Ten equal-count bins of 5 rows: the figures are easy to check by eye."""
    probabilities = np.repeat(np.arange(1, 11) / 20.0, 5)  # 0.05 x5, 0.10 x5 ... 0.50 x5
    labels = np.zeros(50)
    labels[[0, 1]] = 1  # first bin: 2 of 5 observed
    labels[[49]] = 1  # last bin: 1 of 5 observed

    bins = calibration_bins(labels, probabilities, n_bins=10)

    assert [entry["count"] for entry in bins] == [5] * 10
    assert bins[0]["mean_predicted"] == pytest.approx(0.05)
    assert bins[0]["observed_rate"] == pytest.approx(0.4)
    assert bins[9]["mean_predicted"] == pytest.approx(0.50)
    assert bins[9]["observed_rate"] == pytest.approx(0.2)
    assert bins[4]["observed_rate"] == 0.0


def test_feature_importances_are_a_share_of_the_total(trained_small):
    importances = trained_small.metrics["feature_importances"]

    assert sum(importances.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(value >= 0 for value in importances.values())


def test_the_xgboost_parameters_are_the_documented_ones(trained_small):
    assert trained_small.metrics["xgboost"] == {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
        "random_state": SMALL_SEED,
        "tree_method": "hist",
        "n_jobs": 1,
    }


def test_the_calibration_method_is_chosen_by_brier_among_the_eligible_and_recorded(trained_small):
    calibration = trained_small.metrics["calibration"]

    assert calibration["method"] in {"isotonic", "sigmoid"}
    briers = calibration["cross_validated_brier"]
    extremes = calibration["out_of_fold_extreme_outputs"]
    assert set(briers) == set(extremes) == {"isotonic", "sigmoid"}
    eligible = [method for method in briers if extremes[method] == 0] or list(briers)
    assert calibration["method"] == min(eligible, key=briers.get)


def _calibrated_world(seed: int = 0):
    """Three samples of one world whose true probability is a known function of one score, and a
    classifier fitted on the first: already close to calibrated, so a calibrator should barely move it."""
    from xgboost import XGBClassifier

    rng = np.random.default_rng(seed)

    def sample(n: int):
        score = rng.normal(0.0, 1.5, size=n)
        probability = 1.0 / (1.0 + np.exp(-(score - 2.0)))  # mean about 0.15, down to under 0.01
        return pd.DataFrame({"score": score}), (rng.random(n) < probability).astype(int)

    (x_fit, y_fit), (x_cal, y_cal), (x_test, _) = sample(8000), sample(4000), sample(4000)
    xgb = XGBClassifier(n_estimators=60, max_depth=2, learning_rate=0.1, random_state=0, tree_method="hist", n_jobs=1)
    return LogOddsClassifier(xgb.fit(x_fit, y_fit)), x_cal, y_cal, x_test


def test_the_calibrator_is_fitted_to_log_odds_and_barely_moves_a_model_that_is_already_calibrated():
    """scikit-learn fits a sigmoid to decision_function when there is one and to the probability itself
    when there is not. On the probability it cannot output below about 0.06; on the log-odds it is
    close to the identity for a model that was calibrated to begin with."""
    base, x_cal, y_cal, x_test = _calibrated_world()

    calibrated = calibrate(base, x_cal, y_cal, "sigmoid")
    raw = base.predict_proba(x_test)[:, 1]
    served = calibrated.predict_proba(x_test)[:, 1]

    assert raw.min() < 0.03
    assert served.min() < 0.04
    assert np.abs(served - raw).max() < 0.05


def test_the_probability_space_calibrator_is_the_defect_this_guards_against():
    """The same calibration through the bare classifier (no decision_function) floors the output."""
    base, x_cal, y_cal, x_test = _calibrated_world()

    bare = calibrate(base.estimator, x_cal, y_cal, "sigmoid")

    assert bare.predict_proba(x_test)[:, 1].min() > 0.05


def test_the_trained_model_keeps_the_raw_models_low_end(trained_small):
    """On the small run too: the calibrated model's lowest probability stays near the raw model's."""
    test = trained_small.metrics["test"]
    raw = trained_small.metrics["uncalibrated_test"]

    assert test["min_probability"] <= raw["min_probability"] + 0.02
    assert abs(test["brier"] - raw["brier"]) <= 0.002


def _step_problem():
    """Labels whose true rate is a step in the score: isotonic follows the steps (better Brier) and
    memorises runs of all-0 and all-1 rows, while a sigmoid cannot reach exactly 0 or 1."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    features = pd.DataFrame({"score": rng.normal(size=600)})
    score = features["score"].to_numpy()
    rate = np.where(score < -0.5, 0.03, np.where(score > 0.5, 0.97, 0.5))
    labels = (rng.random(len(score)) < rate).astype(int)
    base = LogisticRegression().fit(features, labels)
    groups = np.repeat(np.arange(60), 10)
    return base, features, labels, groups


def test_isotonic_is_not_chosen_when_it_outputs_exactly_0_or_1_even_with_the_better_brier():
    base, features, labels, groups = _step_problem()

    method, details = choose_calibration(base, features, labels, groups)

    briers, extremes = details["cross_validated_brier"], details["out_of_fold_extreme_outputs"]
    assert briers["isotonic"] < briers["sigmoid"]  # isotonic would win on Brier alone
    assert extremes["isotonic"] > 0
    assert extremes["sigmoid"] == 0
    assert method == "sigmoid"


# --- the artifact -------------------------------------------------------------------------------


def test_the_artifact_name_is_the_date_and_the_hash_of_its_bytes(trained_small):
    path = trained_small.artifact_path
    match = ARTIFACT_NAME.match(path.name)

    assert match, path.name
    assert match.group(1) == TEST_TRAIN_DATE.strftime("%Y%m%d")
    assert match.group(2) == hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def test_the_metrics_name_the_artifact_and_the_version(trained_small):
    assert trained_small.metrics["artifact"] == trained_small.artifact_path.name
    assert trained_small.metrics["model_version"] == trained_small.artifact_path.stem
    assert trained_small.model_version == trained_small.artifact_path.stem


def test_the_default_date_is_today(tmp_path):
    result = train(seed=3, students=200, out_dir=tmp_path)

    assert result.artifact_path.name.startswith(f"noshow-{date.today():%Y%m%d}-")


def test_training_leaves_no_other_files(trained_small):
    names = sorted(path.name for path in trained_small.artifact_path.parent.iterdir())

    assert names == sorted(["metrics.json", trained_small.artifact_path.name])


# --- the command line ---------------------------------------------------------------------------


def test_the_command_line_trains_and_reports(tmp_path, capsys):
    code = main(["--seed", "5", "--students", "200", "--out", str(tmp_path)])

    assert code == 0
    written = sorted(path.name for path in tmp_path.iterdir())
    assert "metrics.json" in written
    assert any(ARTIFACT_NAME.match(name) for name in written)
    output = capsys.readouterr().out
    assert "noshow-" in output
    assert "roc_auc" in output.lower() or "auc" in output.lower()


def test_the_command_line_defaults_are_the_documented_seed_and_size():
    args = train_module.parse_args([])

    assert args.seed == 42
    assert args.students == 2000
    assert str(args.out).replace("\\", "/").endswith("app/ml/artifacts")
