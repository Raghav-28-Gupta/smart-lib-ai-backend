"""A seeded synthetic booking simulator: the training data for the no-show model.

`simulate(n_students, seed)` returns one row per booking with the ten features (FEATURE_ORDER), the
label `no_show` (0 or 1) and `student_id`. It runs in two stages so each can be tested on its own:

1. `generate_bookings` draws the world with `numpy.random.default_rng(seed)`: every student has a latent
   reliability (Beta distributed) and a booking rate (Gamma distributed), and makes a stream of seat
   (70 percent) and room (30 percent) bookings over a window of 120 days. The probability of a no-show
   is a logistic of the student's latent reliability, the lead time (a longer lead raises it), a late
   evening or night hour (raises it), room versus seat (rooms are missed more) and a small noise term;
   the label is drawn from it. The constants below are calibrated so the overall rate sits near 15
   percent (the spec asks for 12 to 18); tests check it for several seeds.
2. `build_features` turns those bookings into features by calling `features_from_history` (history.py,
   the same code the Node backend mirrors) for each booking, at the moment it was requested, on that
   student's earlier bookings only. A booking's outcome is visible to a later request only once the
   booking has started: a booking requested earlier but starting later is shown as a plain `booked`
   row, which counts toward `bookings_last_30d` and nothing else. `reliability_score` is rebuilt from
   those same known outcomes (start 100, +2 per attended up to 100, -10 per miss down to 0), never from
   the latent value, so the model cannot see what it is meant to infer.

The simulated calendar starts on a fixed Monday in the library's time zone, so a seed alone decides
the data (no wall clock). Everything the model sees at serving time is rebuilt here with the same code.
"""

from bisect import bisect_left
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.ml.noshow.features import FEATURE_ORDER
from app.ml.noshow.history import LIBRARY_TZ, features_from_history

WINDOW_DAYS = 120
# Day 0 of the simulated calendar: a Monday, midnight in the library's zone. Arbitrary but fixed.
SIM_ORIGIN = datetime(2026, 1, 5, tzinfo=ZoneInfo(LIBRARY_TZ))
_ORIGIN_SECONDS = int(SIM_ORIGIN.timestamp())

# Students: latent reliability r ~ Beta(a, b) (mean 0.83); bookings per day ~ Gamma (mean 0.12, so
# about 14 bookings in the window, with a long tail of heavy users). Every student books at least once.
RELIABILITY_ALPHA = 8.0
RELIABILITY_BETA = 1.6
RATE_SHAPE = 2.0
RATE_SCALE = 0.06

SHARE_ROOMS = 0.30
SEAT_DURATIONS = ([60, 90, 120, 150, 180], [0.15, 0.20, 0.30, 0.20, 0.15])
ROOM_DURATIONS = ([60, 90, 120, 150, 180], [0.25, 0.30, 0.30, 0.10, 0.05])

# Start hour in the library zone (0-23): daytime heavy, an evening tail, almost nothing at night.
_HOUR_WEIGHTS = np.array(
    [0.05, 0.03, 0.02, 0.02, 0.02, 0.03, 0.05, 2, 6, 9, 10, 10, 7, 8, 10, 10, 9, 7, 6, 5, 4, 3, 2, 1],
    dtype=float,
)
HOUR_PROBABILITIES = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()
# Monday to Sunday: quieter weekends.
_WEEKDAY_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.35])
_MINUTES = np.array([0, 15, 30, 45])
_MINUTE_PROBABILITIES = np.array([0.4, 0.1, 0.4, 0.1])

# Lead time: 15 percent walk-ups (minutes), the rest lognormal with a median of 10 hours; clipped to
# 5 minutes ... 14 days and rounded to whole minutes.
WALK_UP_SHARE = 0.15
WALK_UP_MEAN_HOURS = 0.5
LEAD_MEDIAN_HOURS = 10.0
LEAD_SIGMA = 1.3
MIN_LEAD_MINUTES = 5
MAX_LEAD_HOURS = 14 * 24

# The no-show logit: z = INTERCEPT + RELIABILITY_COEF * (1 - r) + LEAD_COEF * lead_scale + LATE_COEF * late
# + ROOM_COEF * room + Normal(0, NOISE_SD), where lead_scale = log1p(lead hours) / log1p(168) (about 1
# at a week) and late is 1 for starts at 20:00 or later and before 06:00.
INTERCEPT = -5.0
RELIABILITY_COEF = 9.0
LEAD_COEF = 2.4
LATE_COEF = 1.1
ROOM_COEF = 0.5
NOISE_SD = 0.3
_LEAD_SCALE_HOURS = 168.0
LATE_FROM_HOUR = 20
LATE_BEFORE_HOUR = 6

_FEATURE_DTYPES = {
    "hour_of_day": "int64",
    "day_of_week": "int64",
    "duration_minutes": "int64",
    "lead_time_hours": "float64",
    "past_attendance_ratio": "float64",
    "consecutive_attended_count": "int64",
    "recent_noshows_count": "int64",
    "reliability_score": "float64",
    "bookings_last_30d": "int64",
    "no_show": "int64",
    "student_id": "int64",
}

_INITIAL_RELIABILITY = 100.0
_ATTENDED_BONUS = 2.0
_MISS_PENALTY = 10.0


def _check_students(n_students: int) -> None:
    if isinstance(n_students, bool) or not isinstance(n_students, (int, np.integer)) or n_students < 1:
        raise ValueError(f"n_students must be a positive integer, got {n_students!r}")


def _check_seed(seed: int) -> None:
    # default_rng(None) would quietly give a different world on every call, so it is refused here.
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed!r}")


def generate_bookings(n_students: int, seed: int) -> pd.DataFrame:
    """Draw the students and their bookings. Sorted by student, then by when each booking was requested.

    Columns: student_id (int), requested_at / start / end (UTC timestamps), resource_type ("seat" or
    "room"), no_show (0/1 label), plus two diagnostics the features never see: latent_reliability and
    p_no_show (the probability the label was drawn from).
    """
    _check_students(n_students)
    _check_seed(seed)
    rng = np.random.default_rng(seed)

    latent = rng.beta(RELIABILITY_ALPHA, RELIABILITY_BETA, size=n_students)
    rate_per_day = rng.gamma(RATE_SHAPE, RATE_SCALE, size=n_students)
    counts = 1 + rng.poisson(rate_per_day * WINDOW_DAYS)
    total = int(counts.sum())
    student = np.repeat(np.arange(n_students), counts)
    reliability = latent[student]

    # When the booking starts, in the library's local time (day index 0 is a Monday).
    day_probabilities = _WEEKDAY_WEIGHTS[np.arange(WINDOW_DAYS) % 7]
    day_probabilities = day_probabilities / day_probabilities.sum()
    day = rng.choice(WINDOW_DAYS, size=total, p=day_probabilities)
    hour = rng.choice(24, size=total, p=HOUR_PROBABILITIES)
    minute = rng.choice(_MINUTES, size=total, p=_MINUTE_PROBABILITIES)
    start_s = _ORIGIN_SECONDS + day * 86400 + hour * 3600 + minute * 60

    # What it is (rooms are 30 percent) and how long it lasts.
    is_room = rng.random(total) < SHARE_ROOMS
    seat_minutes = rng.choice(SEAT_DURATIONS[0], size=total, p=SEAT_DURATIONS[1])
    room_minutes = rng.choice(ROOM_DURATIONS[0], size=total, p=ROOM_DURATIONS[1])
    duration_min = np.where(is_room, room_minutes, seat_minutes)
    end_s = start_s + duration_min * 60

    # How far ahead it was requested.
    walk_up = rng.random(total) < WALK_UP_SHARE
    walk_up_hours = MIN_LEAD_MINUTES / 60 + rng.exponential(WALK_UP_MEAN_HOURS, size=total)
    planned_hours = rng.lognormal(np.log(LEAD_MEDIAN_HOURS), LEAD_SIGMA, size=total)
    lead_hours = np.minimum(np.where(walk_up, walk_up_hours, planned_hours), MAX_LEAD_HOURS)
    lead_min = np.maximum(np.round(lead_hours * 60), MIN_LEAD_MINUTES).astype(np.int64)
    requested_s = start_s - lead_min * 60

    # Will they turn up?
    lead_scale = np.log1p(np.minimum(lead_min / 60.0, _LEAD_SCALE_HOURS)) / np.log1p(_LEAD_SCALE_HOURS)
    late = (hour >= LATE_FROM_HOUR) | (hour < LATE_BEFORE_HOUR)
    logit = (
        INTERCEPT
        + RELIABILITY_COEF * (1.0 - reliability)
        + LEAD_COEF * lead_scale
        + LATE_COEF * late
        + ROOM_COEF * is_room
        + rng.normal(0.0, NOISE_SD, size=total)
    )
    p_no_show = 1.0 / (1.0 + np.exp(-logit))
    no_show = (rng.random(total) < p_no_show).astype(np.int64)

    order = np.lexsort((requested_s, student))

    def moment(seconds: np.ndarray) -> pd.Series:
        return pd.Series(pd.to_datetime(seconds[order], unit="s", utc=True))

    frame = pd.DataFrame(
        {
            "student_id": student[order],
            "requested_at": moment(requested_s),
            "start": moment(start_s),
            "end": moment(end_s),
            "resource_type": np.where(is_room, "room", "seat")[order],
            "no_show": no_show[order],
            "latent_reliability": reliability[order],
            "p_no_show": p_no_show[order],
        }
    )
    return frame


def _datetimes(column: pd.Series) -> list[datetime]:
    return [moment.to_pydatetime() for moment in column]


def build_features(bookings: pd.DataFrame) -> pd.DataFrame:
    """The ten features, the label and the student for each booking, from earlier bookings only.

    `bookings` needs student_id, requested_at, start, end (aware timestamps), resource_type and no_show;
    the result has the same index. For each booking the student's other bookings are shown as they stood
    at the moment it was requested: those that had already started carry their outcome (`no_show` or
    `completed`), those requested earlier but not yet started are plain `booked` rows, and nothing
    requested later is visible.
    """
    columns = [*FEATURE_ORDER, "no_show", "student_id"]
    if len(bookings) == 0:
        return pd.DataFrame({name: [] for name in columns}, index=bookings.index).astype(_FEATURE_DTYPES)

    requested = _datetimes(bookings["requested_at"])
    start = _datetimes(bookings["start"])
    end = _datetimes(bookings["end"])
    kind = bookings["resource_type"].tolist()
    no_show = [int(value) for value in bookings["no_show"]]
    student = bookings["student_id"].to_numpy()

    by_student: dict[Any, list[int]] = {}
    for position, student_id in enumerate(student):
        by_student.setdefault(student_id, []).append(position)

    rows: list[dict[str, Any] | None] = [None] * len(bookings)
    for positions in by_student.values():
        positions.sort(key=lambda p: requested[p])  # stable: ties keep their input order
        known = [
            {
                "status": "no_show" if no_show[p] else "completed",
                "startTime": start[p],
                "requestedAt": requested[p],
                "overturned": False,
            }
            for p in positions
        ]
        pending = [{**row, "status": "booked"} for row in known]

        # The reliability score after each outcome, in the order the bookings started.
        started = sorted(range(len(positions)), key=lambda k: start[positions[k]])
        started_at = [start[positions[k]] for k in started]
        score = _INITIAL_RELIABILITY
        score_after = [score]
        for k in started:
            if no_show[positions[k]]:
                score = max(0.0, score - _MISS_PENALTY)
            else:
                score = min(_INITIAL_RELIABILITY, score + _ATTENDED_BONUS)
            score_after.append(score)

        for i, p in enumerate(positions):
            now = requested[p]
            visible = [known[j] if start[positions[j]] < now else pending[j] for j in range(i)]
            known_outcomes = bisect_left(started_at, now)  # bookings that started before this request
            features = features_from_history(
                {"startTime": start[p], "endTime": end[p], "requestedAt": now, "resourceType": kind[p]},
                visible,
                score_after[known_outcomes],
                now,
                LIBRARY_TZ,
            )
            rows[p] = {**features, "no_show": no_show[p], "student_id": student[p]}

    return pd.DataFrame(rows, index=bookings.index, columns=columns).astype(_FEATURE_DTYPES)


def simulate(n_students: int, seed: int) -> pd.DataFrame:
    """One row per simulated booking: the ten features, `no_show` (0/1) and `student_id`."""
    return build_features(generate_bookings(n_students, seed)).reset_index(drop=True)
