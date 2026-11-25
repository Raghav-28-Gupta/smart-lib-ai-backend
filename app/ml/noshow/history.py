"""History to features: the ten no-show features, computed from a student's other bookings.

This is the Python side of the contract in spec 2026-11-24-no-show-handling, section 6. The Node
backend builds the same features from its database in TypeScript, and the simulator calls this very
function to build its training rows, so what the model is trained on is what it is later asked about.
tests/fixtures/noshow_features.json pins the two implementations to the same definitions (the backend
tests its builder against a byte-for-byte copy).

Definitions, for a booking being scored at time `now` (the booking is not in `history`):

* A history row is *attended* when its startTime is inside the 90 days up to `now`
  (`now - 90 days <= startTime < now`) and its status is `checked_in` or `completed`, or it is a
  `released` row whose no-show was overturned (`overturned` true: the student was there). It is
  *missed* when it is inside the same window and its status is `no_show`. Every other row (legacy
  `released`, `booked`, `requested`, `cancelled`, `declined`, `expired`) is ignored for attendance.
* past_attendance_ratio = attended / (attended + missed), or 0.85 when there are none of either.
* consecutive_attended_count = attended rows that start after the most recent missed row (all of
  them when there is no miss), at most 20. A row that starts at the very same instant as that miss is
  not "after" it.
* recent_noshows_count = missed rows with `now - 30 days <= startTime < now`, at most 10.
* bookings_last_30d = rows with `now - 30 days <= requestedAt <= now` whose status is not
  `cancelled`, `declined` or `expired`, at most 30.
* reliability_score is passed through; lead_time_hours = max(0, start - requestedAt) in hours;
  duration_minutes = whole minutes from start to end; hour_of_day and day_of_week come from the start
  time in the library's time zone, Monday = 0.
"""

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.ml.noshow.features import FEATURE_ORDER

LIBRARY_TZ = "Asia/Kolkata"

PRIOR_ATTENDANCE_RATIO = 0.85
HISTORY_DAYS = 90
RECENT_DAYS = 30
MAX_CONSECUTIVE_ATTENDED = 20
MAX_RECENT_NOSHOWS = 10
MAX_BOOKINGS_LAST_30D = 30

_ATTENDED_STATUSES = frozenset({"checked_in", "completed"})
_NOT_A_BOOKING_STATUSES = frozenset({"cancelled", "declined", "expired"})
_RESOURCE_TYPES = frozenset({"seat", "room"})


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")


def features_from_history(
    booking: dict[str, Any],
    history: list[dict[str, Any]],
    reliability_score: float,
    now: datetime,
    tz: str = LIBRARY_TZ,
) -> dict[str, Any]:
    """The ten features (keys in FEATURE_ORDER) for `booking`, given the student's other bookings.

    booking: {"startTime", "endTime", "requestedAt", "resourceType"} (aware datetimes; "seat" or "room").
    history: [{"status", "startTime", "requestedAt", "overturned"?}, ...] in any order; `overturned` is
        true only for a `released` row whose no-show was overturned and may be missing (false).
    """
    start: datetime = booking["startTime"]
    end: datetime = booking["endTime"]
    requested_at: datetime = booking["requestedAt"]
    resource_type = booking["resourceType"]
    _require_aware(now, "now")
    _require_aware(start, "booking startTime")
    _require_aware(end, "booking endTime")
    _require_aware(requested_at, "booking requestedAt")
    if resource_type not in _RESOURCE_TYPES:
        raise ValueError(f"booking resourceType must be 'seat' or 'room', got {resource_type!r}")

    history_from = now - timedelta(days=HISTORY_DAYS)
    recent_from = now - timedelta(days=RECENT_DAYS)

    attended_starts: list[datetime] = []
    missed = 0
    recent_noshows = 0
    last_miss: datetime | None = None
    bookings_last_30d = 0

    for row in history:
        status = row["status"]
        row_start: datetime = row["startTime"]
        row_requested: datetime = row["requestedAt"]
        _require_aware(row_start, "history startTime")
        _require_aware(row_requested, "history requestedAt")

        if status not in _NOT_A_BOOKING_STATUSES and recent_from <= row_requested <= now:
            bookings_last_30d += 1

        if not (history_from <= row_start < now):
            continue
        if status == "no_show":
            missed += 1
            if row_start >= recent_from:
                recent_noshows += 1
            if last_miss is None or row_start > last_miss:
                last_miss = row_start
        elif status in _ATTENDED_STATUSES or (status == "released" and row.get("overturned", False)):
            attended_starts.append(row_start)

    attended = len(attended_starts)
    if attended + missed == 0:
        ratio = PRIOR_ATTENDANCE_RATIO
    else:
        ratio = attended / (attended + missed)
    if last_miss is None:
        consecutive = attended
    else:
        consecutive = sum(1 for s in attended_starts if s > last_miss)

    local_start = start.astimezone(ZoneInfo(tz))
    features: dict[str, Any] = {
        "hour_of_day": local_start.hour,
        "day_of_week": local_start.weekday(),
        "duration_minutes": int(round((end - start).total_seconds() / 60)),
        "lead_time_hours": max(0.0, (start - requested_at).total_seconds() / 3600),
        "resource_type": resource_type,
        "past_attendance_ratio": float(ratio),
        "consecutive_attended_count": min(consecutive, MAX_CONSECUTIVE_ATTENDED),
        "recent_noshows_count": min(recent_noshows, MAX_RECENT_NOSHOWS),
        "reliability_score": float(reliability_score),
        "bookings_last_30d": min(bookings_last_30d, MAX_BOOKINGS_LAST_30D),
    }
    return {name: features[name] for name in FEATURE_ORDER}
