"""The no-show model's input: ten features in a fixed order, and the interface a model offers.

The Node backend computes the features from its database (spec 2026-11-24-no-show-handling,
section 6) and sends them; this service never reads a database. The order below is the column
order the model is trained and queried in.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.schemas.noshow import NoShowRequest

FEATURE_ORDER: list[str] = [
    "hour_of_day",
    "day_of_week",
    "duration_minutes",
    "lead_time_hours",
    "resource_type",
    "past_attendance_ratio",
    "consecutive_attended_count",
    "recent_noshows_count",
    "reliability_score",
    "bookings_last_30d",
]

# Fairness bound: the largest penalty the service ever returns. The Node side clamps again.
MAX_PENALTY_CAP: float = 0.2

# How the one text feature becomes a number. Training and serving both go through encode_features
# below, and a saved model records this mapping and refuses to load if it ever changes.
RESOURCE_TYPE_CODES: dict[str, int] = {"seat": 0, "room": 1}


class NoShowModel(Protocol):
    """What the endpoint needs from a loaded model (the trained one implements it in Task 13)."""

    def predict(self, features: dict[str, Any]) -> tuple[float, float, str]:
        """Return (probability of a no-show, penalty factor, model version) for one booking."""
        ...


def from_request(request: "NoShowRequest") -> dict[str, Any]:
    """The request's ten features as a dict, keys in FEATURE_ORDER."""
    return {name: getattr(request, name) for name in FEATURE_ORDER}


def encode_features(data: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """The model's input matrix: FEATURE_ORDER columns, all float64, `resource_type` as 0 / 1.

    The ONE place raw features become numbers. Training passes it a frame of many bookings, serving
    passes it one booking as a dict; both get the same columns, order, dtype and encoding, so the two
    paths cannot drift. Extra keys or columns (the label, the student) are ignored. A missing feature,
    an unknown `resource_type` or a value that is not a finite number raises ValueError.
    """
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, Mapping):
        try:
            frame = pd.DataFrame([dict(data)])
        except OverflowError as error:  # an integer beyond float64 (pandas converts it on the way in)
            raise ValueError("every feature must be a finite number") from error
    else:
        raise TypeError(f"expected a mapping or a DataFrame, got {type(data).__name__}")

    missing = [name for name in FEATURE_ORDER if name not in frame.columns]
    if missing:
        raise ValueError(f"missing features: {', '.join(missing)}")

    codes = frame["resource_type"].map(RESOURCE_TYPE_CODES)
    if codes.isna().any():
        raise ValueError(f"resource_type must be one of {sorted(RESOURCE_TYPE_CODES)}")

    encoded = frame[FEATURE_ORDER].copy()
    encoded["resource_type"] = codes
    try:
        encoded = encoded.astype("float64")
    except (TypeError, ValueError, OverflowError) as error:  # OverflowError: an integer beyond float64
        raise ValueError("every feature must be a number") from error
    if not np.isfinite(encoded.to_numpy()).all():
        raise ValueError("every feature must be a finite number")
    return encoded
