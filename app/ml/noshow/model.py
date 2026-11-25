"""The trained no-show model at serving time: pick the artifact, check it, answer one booking.

The artifact is a joblib file in `app/ml/artifacts/`, committed to the repository and named
`noshow-<yyyymmdd>-<hash8>.joblib`: the day it was trained and the first 8 hex characters of the
sha256 of its bytes. That name, without `.joblib`, is the `model_version` every response carries.

Loading uses joblib, which unpickles, so it only ever reads the repository's own artifacts directory
(`ARTIFACTS_DIR`); no path ever comes from a request. As a second guard the bytes are checked against
the hash in the file name before anything is unpickled, so a truncated or altered file is refused.
"""

import hashlib
import io
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from app.ml.noshow.features import (
    FEATURE_ORDER,
    MAX_PENALTY_CAP,
    RESOURCE_TYPE_CODES,
    encode_features,
)

ARTIFACTS_DIR: Path = Path(__file__).resolve().parents[1] / "artifacts"

# The fairness bound (spec section 6): penalty = min(P * ALPHA, MAX_PENALTY_CAP), never negative.
ALPHA: float = 0.5

# A fixed, in-range booking (a seat on a Wednesday morning by a reliable student) used for the one
# test prediction made when an artifact is loaded.
PROBE_FEATURES: dict[str, Any] = {
    "hour_of_day": 10,
    "day_of_week": 2,
    "duration_minutes": 120,
    "lead_time_hours": 24.0,
    "resource_type": "seat",
    "past_attendance_ratio": 0.9,
    "consecutive_attended_count": 5,
    "recent_noshows_count": 0,
    "reliability_score": 90.0,
    "bookings_last_30d": 4,
}

ARTIFACT_FORMAT = 1
ARTIFACT_NAME = re.compile(r"^noshow-(\d{8})-([0-9a-f]{8})\.joblib$")


class LogOddsClassifier(ClassifierMixin, BaseEstimator):
    """A fitted XGBClassifier that also offers its log-odds, so scikit-learn calibrates on those.

    scikit-learn's calibrators ask a classifier for `decision_function` first and use `predict_proba`
    only when there is none; `XGBClassifier` has none. Platt's sigmoid is meant for log-odds, and
    fitted to the probability itself it can only output between about 0.06 and 0.95, which put a floor
    under the risk of the most reliable students. `decision_function` here is XGBoost's raw margin,
    the log-odds of the no-show probability.

    This lives in an app module, not in a script or a test, so a pickled artifact can import it.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.classes_ = np.asarray(estimator.classes_)

    def fit(self, X: Any, y: Any = None) -> "LogOddsClassifier":
        # scikit-learn's check_is_fitted wants a `fit` attribute on anything it freezes; the wrapped
        # classifier is fitted before it is wrapped, and a frozen estimator is never refitted.
        raise NotImplementedError("fit the XGBClassifier first, then wrap it")

    def decision_function(self, X: Any) -> np.ndarray:
        return self.estimator.predict(X, output_margin=True)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator.predict(X)


class ModelLoadError(Exception):
    """The service has no usable model (the router answers 503 for it)."""


class NoArtifactError(ModelLoadError):
    """The artifacts directory holds no artifact."""


class ArtifactError(ModelLoadError):
    """An artifact exists but cannot be used: altered, unreadable, or built for other features."""


def penalty_factor(p: float, alpha: float = ALPHA, cap: float = MAX_PENALTY_CAP) -> float:
    """The bounded priority penalty for a no-show probability: min(max(p * alpha, 0), cap).

    Always within 0 and `cap` (0.2 by default), and never decreasing in `p`. A probability, alpha or
    cap that is not finite raises ValueError rather than producing a penalty from garbage.
    """
    for name, value in (("p", p), ("alpha", alpha), ("cap", cap)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    return min(max(p * alpha, 0.0), cap)


def _artifact_order(path: Path) -> tuple[str, int, str]:
    # Newest first by the date in the name; one day's artifacts by file time, then by name. A git
    # checkout gives every file the same time, so the name's date is what decides in practice.
    match = ARTIFACT_NAME.match(path.name)
    assert match is not None
    return match.group(1), path.stat().st_mtime_ns, path.name


def find_newest_artifact(directory: Path | str) -> Path | None:
    """The newest `noshow-<yyyymmdd>-<hash8>.joblib` in `directory`, or None.

    Newest means the latest date in the file name; artifacts from the same day are ordered by file
    modification time, then by name, so the choice never depends on directory listing order. Anything
    else in the directory (`metrics.json`, a name with a bad date or hash, a folder) is ignored.
    """
    folder = Path(directory)
    candidates: list[Path] = []
    try:
        entries = list(folder.iterdir())
    except OSError:
        return None
    for path in entries:
        match = ARTIFACT_NAME.match(path.name)
        if match is None or not path.is_file():
            continue
        try:
            datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=_artifact_order)


class NoShowModel:
    """A calibrated classifier plus its version; implements features.NoShowModel (the Protocol)."""

    def __init__(self, calibrated: Any, version: str) -> None:
        self.calibrated = calibrated
        self.version = version
        # Requests run in a thread pool; one lock keeps inference trivially thread safe.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, directory: Path | str | None = None) -> "NoShowModel":
        """Load the newest artifact in `directory` (default: the repository's artifacts directory).

        Raises NoArtifactError when there is none and ArtifactError when the newest one is unusable.
        """
        folder = ARTIFACTS_DIR if directory is None else Path(directory)
        path = find_newest_artifact(folder)
        if path is None:
            raise NoArtifactError("no model artifact found")
        match = ARTIFACT_NAME.match(path.name)
        assert match is not None
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ArtifactError("the model artifact could not be read") from error
        if hashlib.sha256(data).hexdigest()[:8] != match.group(2):
            raise ArtifactError("the model artifact does not match the hash in its file name")
        try:
            payload = joblib.load(io.BytesIO(data))
        except Exception as error:  # a corrupt pickle can raise almost anything
            raise ArtifactError("the model artifact could not be unpickled") from error
        return cls.from_payload(payload, path.stem)

    @classmethod
    def from_payload(cls, payload: Any, version: str) -> "NoShowModel":
        if not isinstance(payload, dict) or "model" not in payload:
            raise ArtifactError("the model artifact is not in the expected format")
        if payload.get("feature_order") != FEATURE_ORDER:
            raise ArtifactError("the model artifact was trained on a different feature order")
        if payload.get("encoding") != {"resource_type": RESOURCE_TYPE_CODES}:
            raise ArtifactError("the model artifact used a different feature encoding")
        if not hasattr(payload["model"], "predict_proba"):
            raise ArtifactError("the model artifact holds no classifier")
        model = cls(payload["model"], version)
        # One test prediction through the whole predict path, so a model that unpickles but cannot
        # answer is refused here (the service starts without it and answers 503) instead of raising
        # inside every request.
        try:
            model.predict(PROBE_FEATURES)
        except Exception as error:
            raise ArtifactError("the model artifact failed its test prediction") from error
        return model

    def predict(self, features: dict[str, Any]) -> tuple[float, float, str]:
        """(probability of a no-show in [0, 1], bounded penalty factor, model version) for one booking."""
        row = encode_features(features)
        with self._lock:
            probability = float(self.calibrated.predict_proba(row)[0, 1])
        if not math.isfinite(probability):
            raise ValueError("the model produced a probability that is not a number")
        probability = min(max(probability, 0.0), 1.0)
        return probability, penalty_factor(probability), self.version
