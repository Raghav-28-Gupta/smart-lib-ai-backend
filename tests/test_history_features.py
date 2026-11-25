"""features_from_history against the shared fixture (spec 2026-11-24-no-show-handling, section 6).

tests/fixtures/noshow_features.json is the contract with the Node backend: its expected values were
worked out by hand from the spec's definitions, and the backend's TypeScript builder is tested against
a byte-for-byte copy. The cases carry their own `now`, so this is the one place fixed calendar dates
are allowed; the unit tests below derive their dates from the clock.
"""

import copy
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.ml.noshow.features import FEATURE_ORDER
from app.ml.noshow.history import features_from_history

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "noshow_features.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

INT_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "duration_minutes",
    "consecutive_attended_count",
    "recent_noshows_count",
    "bookings_last_30d",
]
FLOAT_FEATURES = [
    "lead_time_hours",
    "past_attendance_ratio",
    "reliability_score",
]
CASE_IDS = [case["name"] for case in CASES]


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _booking(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "startTime": _dt(raw["startTime"]),
        "endTime": _dt(raw["endTime"]),
        "requestedAt": _dt(raw["requestedAt"]),
        "resourceType": raw["resourceType"],
    }


def _history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = []
    for raw in rows:
        row = {
            "status": raw["status"],
            "startTime": _dt(raw["startTime"]),
            "requestedAt": _dt(raw["requestedAt"]),
        }
        if "overturned" in raw:
            row["overturned"] = raw["overturned"]
        parsed.append(row)
    return parsed


def _run(case: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return features_from_history(
        _booking(case["booking"]),
        _history(case["history"]) if history is None else history,
        case["reliability_score"],
        _dt(case["now"]),
        case["tz"],
    )


def _assert_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert list(actual) == FEATURE_ORDER
    for name in INT_FEATURES:
        assert type(actual[name]) is int, f"{name} should be an int, got {type(actual[name])}"
        assert actual[name] == expected[name], name
    for name in FLOAT_FEATURES:
        assert type(actual[name]) is float, f"{name} should be a float, got {type(actual[name])}"
        assert actual[name] == pytest.approx(expected[name], abs=1e-9), name
    assert actual["resource_type"] == expected["resource_type"]


# ---- the fixture itself -------------------------------------------------------------------------


def test_the_fixture_is_a_list_of_at_least_ten_well_formed_cases() -> None:
    assert isinstance(CASES, list)
    assert len(CASES) >= 10
    assert len(set(CASE_IDS)) == len(CASES), "case names must be unique"
    for case in CASES:
        assert set(case) == {
            "name",
            "description",
            "now",
            "tz",
            "booking",
            "history",
            "reliability_score",
            "expected",
        }
        assert case["description"].strip(), f"{case['name']} needs a description"
        assert set(case["booking"]) == {"startTime", "endTime", "requestedAt", "resourceType"}
        for text in [case["now"], *[case["booking"][k] for k in ("startTime", "endTime", "requestedAt")]]:
            assert _dt(text).tzinfo is not None, f"{case['name']}: {text} has no UTC offset"
        for row in case["history"]:
            assert set(row) - {"overturned"} == {"status", "startTime", "requestedAt"}
            assert _dt(row["startTime"]).tzinfo is not None
            assert _dt(row["requestedAt"]).tzinfo is not None


def test_the_fixtures_expected_keys_equal_feature_order() -> None:
    for case in CASES:
        assert sorted(case["expected"]) == sorted(FEATURE_ORDER), case["name"]
        assert len(case["expected"]) == len(FEATURE_ORDER)


def test_the_fixture_pins_every_rule_the_spec_lists() -> None:
    expected = [case["expected"] for case in CASES]
    assert any(e["past_attendance_ratio"] == 0.85 for e in expected), "the no-history prior"
    assert any(e["consecutive_attended_count"] == 20 for e in expected), "the consecutive cap"
    assert any(e["recent_noshows_count"] == 10 for e in expected), "the recent no-show cap"
    assert any(e["bookings_last_30d"] == 30 for e in expected), "the bookings cap"
    assert any(e["lead_time_hours"] == 0.0 for e in expected), "the lead-time clamp"
    assert {e["day_of_week"] for e in expected} >= {0, 6}, "a Monday and a Sunday"
    assert {e["resource_type"] for e in expected} == {"seat", "room"}
    assert any(e["consecutive_attended_count"] == 3 for e in expected), "one miss then three attended"
    assert any(0 < e["past_attendance_ratio"] < 0.85 for e in expected), "a ratio below the prior"
    for e in expected:
        assert 0 <= e["hour_of_day"] <= 23 and 0 <= e["day_of_week"] <= 6


def test_the_fixture_pins_the_requested_at_upper_bound() -> None:
    """A history row requested after `now` (not counted) and one requested exactly at `now` (counted)."""
    after = at_now = False
    for case in CASES:
        now = _dt(case["now"])
        for row in case["history"]:
            requested = _dt(row["requestedAt"])
            after = after or requested > now
            at_now = at_now or requested == now
    assert after, "no case has a history row requested after now"
    assert at_now, "no case has a history row requested exactly at now"


def test_the_fixture_has_a_start_at_local_hour_0() -> None:
    """JavaScript's Intl with hour12 false can answer 24 for midnight; only a local hour 0 catches it."""
    assert any(
        case["expected"]["hour_of_day"] == 0 and case["tz"] == "Asia/Kolkata" for case in CASES
    )


def test_the_fixture_pins_a_same_instant_tie_in_both_list_orders() -> None:
    """A miss and an attended row with the same startTime, listed miss-first and attended-first."""
    orders: set[str] = set()
    for case in CASES:
        positions: dict[datetime, dict[str, list[int]]] = {}
        for index, row in enumerate(case["history"]):
            kind = "miss" if row["status"] == "no_show" else (
                "attended" if row["status"] in ("completed", "checked_in") else "other"
            )
            positions.setdefault(_dt(row["startTime"]), {}).setdefault(kind, []).append(index)
        for kinds in positions.values():
            if "miss" in kinds and "attended" in kinds:
                orders.add("miss first" if min(kinds["miss"]) < min(kinds["attended"]) else "attended first")
    assert orders == {"miss first", "attended first"}


# ---- every fixture case -------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_fixture_case_passes(case: dict[str, Any]) -> None:
    _assert_matches(_run(case), case["expected"])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_order_of_the_history_list_does_not_matter(case: dict[str, Any]) -> None:
    history = _history(case["history"])
    shuffled = history[:]
    random.Random(7).shuffle(shuffled)
    for variant in (list(reversed(history)), shuffled):
        _assert_matches(_run(case, variant), case["expected"])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_inputs_are_not_modified(case: dict[str, Any]) -> None:
    booking = _booking(case["booking"])
    history = _history(case["history"])
    before = copy.deepcopy((booking, history))
    features_from_history(booking, history, case["reliability_score"], _dt(case["now"]), case["tz"])
    assert (booking, history) == before


# ---- behaviour not tied to a fixed date ---------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _simple_booking(now: datetime) -> dict[str, Any]:
    start = now + timedelta(hours=3)
    return {
        "startTime": start,
        "endTime": start + timedelta(minutes=60),
        "requestedAt": now,
        "resourceType": "seat",
    }


def test_the_default_zone_is_the_library_zone_asia_kolkata() -> None:
    now = _utc_now()
    tomorrow = (now + timedelta(days=1)).replace(hour=20, minute=30, second=0)
    booking = {
        "startTime": tomorrow,
        "endTime": tomorrow + timedelta(minutes=90),
        "requestedAt": now,
        "resourceType": "room",
    }
    result = features_from_history(booking, [], 100.0, now)
    # 20:30 UTC is 02:00 the next day in Asia/Kolkata (UTC+5:30, no daylight saving).
    assert result["hour_of_day"] == 2
    assert result["day_of_week"] == (tomorrow.date() + timedelta(days=1)).weekday()
    in_utc = features_from_history(booking, [], 100.0, now, "UTC")
    assert in_utc["hour_of_day"] == 20
    assert in_utc["day_of_week"] == tomorrow.date().weekday()


def test_a_missing_overturned_key_means_not_overturned() -> None:
    now = _utc_now()
    row = {
        "status": "released",
        "startTime": now - timedelta(days=3),
        "requestedAt": now - timedelta(days=4),
    }
    plain = features_from_history(_simple_booking(now), [row], 100.0, now)
    assert plain["past_attendance_ratio"] == 0.85 and plain["consecutive_attended_count"] == 0
    overturned = features_from_history(_simple_booking(now), [{**row, "overturned": True}], 100.0, now)
    assert overturned["past_attendance_ratio"] == 1.0 and overturned["consecutive_attended_count"] == 1
    explicit_false = features_from_history(_simple_booking(now), [{**row, "overturned": False}], 100.0, now)
    assert explicit_false["past_attendance_ratio"] == 0.85


def test_overturned_only_matters_for_released_rows() -> None:
    now = _utc_now()
    rows = [
        {"status": "no_show", "startTime": now - timedelta(days=3), "requestedAt": now - timedelta(days=4), "overturned": True},
        {"status": "cancelled", "startTime": now - timedelta(days=2), "requestedAt": now - timedelta(days=3), "overturned": True},
    ]
    result = features_from_history(_simple_booking(now), rows, 100.0, now)
    assert result["past_attendance_ratio"] == 0.0  # the no-show stays a miss
    assert result["consecutive_attended_count"] == 0
    assert result["recent_noshows_count"] == 1
    assert result["bookings_last_30d"] == 1  # the cancelled row is not a booking


def test_an_integer_reliability_score_comes_back_as_a_float() -> None:
    now = _utc_now()
    result = features_from_history(_simple_booking(now), [], 100, now)
    assert result["reliability_score"] == 100.0 and type(result["reliability_score"]) is float


def test_the_priors_and_caps_are_the_specs_numbers() -> None:
    now = _utc_now()
    base = _simple_booking(now)
    assert features_from_history(base, [], 50.0, now)["past_attendance_ratio"] == 0.85
    many = [
        {"status": "completed", "startTime": now - timedelta(days=1, hours=i), "requestedAt": now - timedelta(days=2, hours=i)}
        for i in range(40)
    ]
    capped = features_from_history(base, many, 50.0, now)
    assert capped["consecutive_attended_count"] == 20
    assert capped["bookings_last_30d"] == 30
    misses = [{**row, "status": "no_show"} for row in many[:15]]
    assert features_from_history(base, misses, 50.0, now)["recent_noshows_count"] == 10


@pytest.mark.parametrize("field", ["startTime", "endTime", "requestedAt"])
def test_a_naive_booking_time_is_refused(field: str) -> None:
    now = _utc_now()
    booking = _simple_booking(now)
    booking[field] = booking[field].replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        features_from_history(booking, [], 100.0, now)


def test_a_naive_now_or_history_time_is_refused() -> None:
    now = _utc_now()
    with pytest.raises(ValueError, match="timezone"):
        features_from_history(_simple_booking(now), [], 100.0, now.replace(tzinfo=None))
    row = {"status": "completed", "startTime": now.replace(tzinfo=None) - timedelta(days=1), "requestedAt": now - timedelta(days=2)}
    with pytest.raises(ValueError, match="timezone"):
        features_from_history(_simple_booking(now), [row], 100.0, now)


def test_an_unknown_resource_type_is_refused() -> None:
    now = _utc_now()
    booking = {**_simple_booking(now), "resourceType": "desk"}
    with pytest.raises(ValueError, match="resourceType"):
        features_from_history(booking, [], 100.0, now)
