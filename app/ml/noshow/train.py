"""Train, calibrate and write the no-show model (spec 2026-11-24-no-show-handling, section 8).

    python -m app.ml.noshow.train --seed 42 [--students 2000] [--out app/ml/artifacts]

The steps, all seeded so one seed always gives the same files:

1. `simulate(students, seed)`: the synthetic bookings (simulate.py), features from `features_from_history`.
2. Split BY STUDENT into train / calibration / test (60 / 20 / 20 percent of the students): a student's
   bookings are never in two sets, so nothing a student did leaks across the split.
3. Fit `XGBClassifier` (150 trees, depth 4, learning rate 0.05, one thread) on the train split.
4. Calibrate the probabilities on the calibration split, through `LogOddsClassifier` so that the sigmoid
   is fitted to log-odds (see model.py). Isotonic or sigmoid (Platt) is chosen by the
   out-of-fold Brier score on the calibration split itself (5 folds grouped by student), because an
   in-sample score always favours isotonic; a method that outputs exactly 0 or 1 out of fold is not
   eligible (see `choose_calibration`). The chosen one is then fitted on the whole calibration split.
5. Evaluate on the test split: ROC AUC, Brier score, log loss, the Brier score of the base-rate
   predictor (the train mean for every row), ten equal-count calibration bins and their mean absolute
   gap, and the feature importances.
6. Write `noshow-<yyyymmdd>-<hash8>.joblib` (hash8: the first 8 hex of the sha256 of the artifact's
   bytes; the date is the training day) and `metrics.json` beside it. Both are committed, so a deploy
   needs no training. `metrics.json` records no timestamp: the same seed gives the same file.
"""

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from app.ml.noshow.features import FEATURE_ORDER, RESOURCE_TYPE_CODES, encode_features
from app.ml.noshow.model import ARTIFACT_FORMAT, ARTIFACTS_DIR, LogOddsClassifier
from app.ml.noshow.simulate import simulate

DEFAULT_SEED = 42
DEFAULT_STUDENTS = 2000

TRAIN_SHARE = 0.6
CALIBRATION_SHARE = 0.2  # the rest, 0.2, is the test split
MIN_STUDENTS = 50
_SPLIT_STREAM = 1  # a separate random stream from the simulator's, same seed

CALIBRATION_METHODS = ("sigmoid", "isotonic")  # a tie in Brier goes to the simpler, first one
CALIBRATION_FOLDS = 5
N_BINS = 10
_DECIMALS = 6


@dataclass(frozen=True)
class TrainingResult:
    artifact_path: Path
    metrics_path: Path
    metrics: dict[str, Any]
    model_version: str


def xgboost_params(seed: int) -> dict[str, Any]:
    """The classifier's settings, exactly as the ml-pipeline skill gives them. One thread, so one
    seed gives one model."""
    return {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
        "random_state": seed,
        "tree_method": "hist",
        "n_jobs": 1,
    }


def split_by_student(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, calibration, test) frames, 60 / 20 / 20 percent of the STUDENTS, every row with its student."""
    students = np.sort(frame["student_id"].unique())
    if len(students) < MIN_STUDENTS:
        raise ValueError(f"need at least {MIN_STUDENTS} students to split three ways, got {len(students)}")
    shuffled = np.random.default_rng([seed, _SPLIT_STREAM]).permutation(students)
    n_train = round(TRAIN_SHARE * len(students))
    n_calibration = round(CALIBRATION_SHARE * len(students))
    groups = (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_calibration],
        shuffled[n_train + n_calibration :],
    )
    train_rows, calibration_rows, test_rows = (frame[frame["student_id"].isin(ids)] for ids in groups)
    return train_rows, calibration_rows, test_rows


def _assert_split_by_student(*splits: pd.DataFrame) -> None:
    seen: set[Any] = set()
    for rows in splits:
        students = set(rows["student_id"])
        if not seen.isdisjoint(students):
            raise RuntimeError("a student appears in more than one of the train / calibration / test sets")
        seen |= students


def calibrate(base: Any, features: pd.DataFrame, labels: np.ndarray, method: str) -> CalibratedClassifierCV:
    """`base` (already fitted) calibrated on (features, labels) with `method`.

    Pass a `LogOddsClassifier`, not the bare XGBClassifier: scikit-learn fits Platt's sigmoid to a
    classifier's `decision_function` when it has one and to the probability itself when it does not.
    """
    return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(features, labels)


def choose_calibration(
    base: Any, features: pd.DataFrame, labels: np.ndarray, groups: np.ndarray
) -> tuple[str, dict[str, Any]]:
    """Pick isotonic or sigmoid for the calibration split.

    Each fold's calibrator is fitted on the other folds (split by student) and scored on the held-out
    one, so the Brier score is out of fold and isotonic gets no credit for memorising the split. A
    method is eligible only if none of its out-of-fold probabilities is exactly 0 or 1: isotonic's
    flat ends can say "never" or "certainly" for a booking that is nothing of the kind, and the
    console would print it. The lowest Brier among the eligible methods wins (if none is eligible, the
    lowest overall). Returns (method, {"cross_validated_brier": ..., "out_of_fold_extreme_outputs": ...}).
    """
    folds = GroupKFold(n_splits=min(CALIBRATION_FOLDS, len(set(groups))))
    briers: dict[str, float] = {}
    extremes: dict[str, int] = {}
    for method in CALIBRATION_METHODS:
        out_of_fold = np.empty(len(labels))
        for fit_rows, held_rows in folds.split(features, labels, groups):
            calibrator = calibrate(base, features.iloc[fit_rows], labels[fit_rows], method)
            out_of_fold[held_rows] = calibrator.predict_proba(features.iloc[held_rows])[:, 1]
        briers[method] = float(brier_score_loss(labels, out_of_fold))
        extremes[method] = int(np.sum((out_of_fold <= 0.0) | (out_of_fold >= 1.0)))
    eligible = [method for method in CALIBRATION_METHODS if extremes[method] == 0] or list(CALIBRATION_METHODS)
    best = min(eligible, key=lambda method: briers[method])
    return best, {"cross_validated_brier": briers, "out_of_fold_extreme_outputs": extremes}


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Equal-count calibration bins: rows sorted by predicted probability and cut into `n_bins` runs.

    Each bin is {mean_predicted, observed_rate, count}. Equal-count (not equal-width) bins keep every
    bin well populated even though most probabilities sit at the low end; ties keep input order.
    """
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities, kind="stable")
    bins = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        bins.append(
            {
                "mean_predicted": float(probabilities[chunk].mean()),
                "observed_rate": float(labels[chunk].mean()),
                "count": int(len(chunk)),
            }
        )
    return bins


def mean_abs_calibration_gap(bins: list[dict[str, Any]]) -> float:
    """The mean over the populated bins of |mean predicted - observed rate|."""
    populated = [entry for entry in bins if entry["count"] > 0]
    return float(np.mean([abs(entry["mean_predicted"] - entry["observed_rate"]) for entry in populated]))


def _scores(labels: np.ndarray, probabilities: np.ndarray, base_rate: float) -> dict[str, float]:
    bins = calibration_bins(labels, probabilities)
    return {
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "base_rate_brier": float(np.mean((base_rate - labels) ** 2)),
        "mean_abs_calibration_gap": mean_abs_calibration_gap(bins),
    }


def _round(value: Any) -> Any:
    """Round every float to a fixed number of decimals and turn numpy scalars into plain ones."""
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), _DECIMALS)
    if isinstance(value, np.integer):
        return int(value)
    return value


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def train(
    seed: int = DEFAULT_SEED,
    students: int = DEFAULT_STUDENTS,
    out_dir: Path | str = ARTIFACTS_DIR,
    train_date: date | None = None,
) -> TrainingResult:
    """Run the whole pipeline and write the artifact and metrics.json into `out_dir`."""
    trained_on = train_date or date.today()
    frame = simulate(students, seed)
    fit_rows, calibration_rows, test_rows = split_by_student(frame, seed)
    _assert_split_by_student(fit_rows, calibration_rows, test_rows)

    x_fit, y_fit = encode_features(fit_rows), fit_rows["no_show"].to_numpy()
    x_calibration, y_calibration = encode_features(calibration_rows), calibration_rows["no_show"].to_numpy()
    x_test, y_test = encode_features(test_rows), test_rows["no_show"].to_numpy()

    params = xgboost_params(seed)
    base = XGBClassifier(**params).fit(x_fit, y_fit)

    # Calibrate through the log-odds wrapper (see LogOddsClassifier), for the choice and the final fit.
    log_odds = LogOddsClassifier(base)
    method, calibration_choice = choose_calibration(
        log_odds, x_calibration, y_calibration, calibration_rows["student_id"].to_numpy()
    )
    calibrated = calibrate(log_odds, x_calibration, y_calibration, method)

    train_mean = float(y_fit.mean())
    calibrated_test = calibrated.predict_proba(x_test)[:, 1]
    raw_test = base.predict_proba(x_test)[:, 1]

    libraries = library_versions()
    payload = {
        "format": ARTIFACT_FORMAT,
        "model": calibrated,
        "feature_order": list(FEATURE_ORDER),
        "encoding": {"resource_type": dict(RESOURCE_TYPE_CODES)},
        "calibration_method": method,
        "seed": seed,
        "trained_on": trained_on.isoformat(),
        "libraries": libraries,
    }
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    data = buffer.getvalue()
    artifact_name = f"noshow-{trained_on:%Y%m%d}-{hashlib.sha256(data).hexdigest()[:8]}.joblib"

    metrics = _round(
        {
            "artifact": artifact_name,
            "model_version": artifact_name.removesuffix(".joblib"),
            "seed": seed,
            "students": {
                "total": int(frame["student_id"].nunique()),
                "train": int(fit_rows["student_id"].nunique()),
                "calibration": int(calibration_rows["student_id"].nunique()),
                "test": int(test_rows["student_id"].nunique()),
            },
            "rows": {
                "total": len(frame),
                "train": len(fit_rows),
                "calibration": len(calibration_rows),
                "test": len(test_rows),
            },
            "no_show_rate": {
                "train": train_mean,
                "calibration": float(y_calibration.mean()),
                "test": float(y_test.mean()),
            },
            "base_rate": {"train_mean": train_mean},
            "xgboost": params,
            "calibration": {
                "method": method,
                **calibration_choice,
                "folds": min(CALIBRATION_FOLDS, calibration_rows["student_id"].nunique()),
            },
            "test": _scores(y_test, calibrated_test, train_mean),
            "uncalibrated_test": _scores(y_test, raw_test, train_mean),
            "calibration_bin_scheme": "equal_count",
            "calibration_bins": calibration_bins(y_test, calibrated_test),
            "feature_importances": dict(zip(FEATURE_ORDER, base.feature_importances_)),
            "libraries": libraries,
        }
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / artifact_name
    metrics_path = out / "metrics.json"
    artifact_path.write_bytes(data)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    return TrainingResult(artifact_path, metrics_path, metrics, metrics["model_version"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.ml.noshow.train",
        description="Train, calibrate and save the no-show model on simulated bookings.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default %(default)s)")
    parser.add_argument(
        "--students", type=int, default=DEFAULT_STUDENTS, help="simulated students (default %(default)s)"
    )
    parser.add_argument(
        "--out", type=Path, default=ARTIFACTS_DIR, help="directory for the artifact and metrics.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    result = train(seed=args.seed, students=args.students, out_dir=args.out)
    seconds = time.perf_counter() - started
    test = result.metrics["test"]
    students, rows = result.metrics["students"]["total"], result.metrics["rows"]["total"]
    print(f"Trained {result.model_version} on {students} students ({rows} bookings) in {seconds:.1f} s")
    print(
        f"  test ROC AUC {test['roc_auc']:.4f}; Brier {test['brier']:.4f} vs base rate "
        f"{test['base_rate_brier']:.4f}; mean calibration gap {test['mean_abs_calibration_gap']:.4f}; "
        f"calibration: {result.metrics['calibration']['method']}"
    )
    print(f"  wrote {result.artifact_path}")
    print(f"  wrote {result.metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
