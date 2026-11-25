"""The synthetic booking simulator (spec 2026-11-24-no-show-handling, section 8).

Everything is seeded; no test touches the network, a clock-dependent value or a trained model.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from app.ml.noshow import simulate as simulate_module
from app.ml.noshow.features import FEATURE_ORDER
from app.ml.noshow.simulate import build_features, generate_bookings, simulate
from app.schemas.noshow import NoShowRequest

SMALL = 300  # students: about 4,500 bookings, well under a second


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve of `score` for `label` (rank-based; ties get average ranks)."""
    ranks = rankdata(score)
    positives = int(label.sum())
    negatives = len(label) - positives
    return float((ranks[label == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return simulate(SMALL, 42)


# ---- shape and determinism ----------------------------------------------------------------------


def test_simulate_is_deterministic_for_a_seed_and_differs_for_another() -> None:
    first = simulate(120, 7)
    again = simulate(120, 7)
    other = simulate(120, 8)
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)


def test_one_row_per_booking_with_the_ten_features_the_label_and_the_student(frame: pd.DataFrame) -> None:
    assert list(frame.columns) == [*FEATURE_ORDER, "no_show", "student_id"]
    assert not frame.isna().any().any()
    assert frame.index.equals(pd.RangeIndex(len(frame)))
    assert set(frame["no_show"].unique()) == {0, 1}
    assert frame["student_id"].nunique() == SMALL, "every student makes at least one booking"
    for name in ["hour_of_day", "day_of_week", "duration_minutes", "consecutive_attended_count",
                 "recent_noshows_count", "bookings_last_30d", "no_show", "student_id"]:
        assert pd.api.types.is_integer_dtype(frame[name]), name
    for name in ["lead_time_hours", "past_attendance_ratio", "reliability_score"]:
        assert pd.api.types.is_float_dtype(frame[name]), name


def test_every_row_satisfies_the_endpoint_contract(frame: pd.DataFrame) -> None:
    for record in frame[FEATURE_ORDER].to_dict("records"):
        NoShowRequest(**record)  # raises on any out-of-range value
    assert frame["consecutive_attended_count"].max() <= 20
    assert frame["recent_noshows_count"].max() <= 10
    assert frame["bookings_last_30d"].max() <= 30


def test_simulate_needs_at_least_one_student() -> None:
    for bad in (0, -3):
        with pytest.raises(ValueError, match="n_students"):
            simulate(bad, 1)


@pytest.mark.parametrize("bad_seed", [None, 1.5, "42", True, [1, 2]])
def test_simulate_needs_an_integer_seed(bad_seed: Any) -> None:
    """numpy would turn seed=None into a different random world on every call; refuse it."""
    with pytest.raises(TypeError, match="seed"):
        simulate(10, bad_seed)


def test_simulate_refuses_a_negative_seed_and_accepts_a_numpy_integer() -> None:
    with pytest.raises(ValueError, match="seed"):
        simulate(10, -1)
    pd.testing.assert_frame_equal(simulate(10, np.int64(3)), simulate(10, 3))


def test_build_features_of_no_bookings_is_an_empty_frame_with_the_usual_columns() -> None:
    out = build_features(generate_bookings(5, 1).iloc[0:0])
    assert len(out) == 0
    assert list(out.columns) == [*FEATURE_ORDER, "no_show", "student_id"]
    for name in ["hour_of_day", "day_of_week", "duration_minutes", "consecutive_attended_count",
                 "recent_noshows_count", "bookings_last_30d", "no_show", "student_id"]:
        assert pd.api.types.is_integer_dtype(out[name]), name
    for name in ["lead_time_hours", "past_attendance_ratio", "reliability_score"]:
        assert pd.api.types.is_float_dtype(out[name]), name


# ---- the distributions the spec asks for --------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 42])
def test_the_base_no_show_rate_is_between_12_and_18_percent(seed: int) -> None:
    rate = simulate(SMALL, seed)["no_show"].mean()
    assert 0.12 <= rate <= 0.18, f"seed {seed}: {rate:.4f}"


def test_the_mix_of_resources_hours_lead_times_and_durations(frame: pd.DataFrame) -> None:
    rooms = (frame["resource_type"] == "room").mean()
    assert 0.26 <= rooms <= 0.34, "about 70 percent seats, 30 percent rooms"
    assert set(frame["resource_type"].unique()) == {"seat", "room"}
    assert frame["duration_minutes"].between(60, 180).all()
    assert frame["duration_minutes"].nunique() >= 4
    daytime = frame["hour_of_day"].between(8, 19).mean()
    evening = (frame["hour_of_day"] >= 20).mean()
    assert daytime >= 0.75, "hours are skewed to daytime"
    assert 0.02 <= evening <= 0.2, "with an evening tail"
    assert frame["lead_time_hours"].min() < 1.0, "lead times start at minutes"
    assert frame["lead_time_hours"].max() > 72.0, "and reach days"
    assert frame["lead_time_hours"].min() >= 5 / 60 - 1e-9
    assert frame["day_of_week"].nunique() == 7


def test_a_long_lead_time_raises_the_no_show_rate_on_average() -> None:
    data = simulate(400, 5)
    long_lead = data[data["lead_time_hours"] >= 48]["no_show"].mean()
    short_lead = data[data["lead_time_hours"] < 6]["no_show"].mean()
    assert long_lead > short_lead + 0.05, (long_lead, short_lead)


def test_late_evening_hours_raise_the_no_show_rate_on_average() -> None:
    data = simulate(400, 5)
    late = data[data["hour_of_day"] >= 20]["no_show"].mean()
    daytime = data[data["hour_of_day"].between(9, 17)]["no_show"].mean()
    assert late > daytime + 0.05, (late, daytime)


def test_rooms_are_missed_more_often_than_seats() -> None:
    rates = simulate(800, 5).groupby("resource_type")["no_show"].mean()
    assert rates["room"] > rates["seat"] + 0.03, rates.to_dict()


def test_the_simulated_student_reliability_explains_the_label_better_than_chance() -> None:
    bookings = generate_bookings(SMALL, 42)
    latent_auc = _auc(-bookings["latent_reliability"].to_numpy(), bookings["no_show"].to_numpy())
    assert latent_auc > 0.62, f"latent reliability AUC {latent_auc:.3f}"

    data = build_features(bookings)
    seen = data[data["past_attendance_ratio"] != 0.85]  # rows with some history to learn from
    observed_auc = _auc(-seen["reliability_score"].to_numpy(), seen["no_show"].to_numpy())
    assert observed_auc > 0.58, f"history-derived reliability_score AUC {observed_auc:.3f}"


# ---- features come from earlier bookings only ---------------------------------------------------


def test_a_new_students_first_booking_uses_the_prior(frame: pd.DataFrame) -> None:
    first = frame.groupby("student_id").head(1)
    assert len(first) == SMALL
    assert (first["past_attendance_ratio"] == 0.85).all()
    assert (first["consecutive_attended_count"] == 0).all()
    assert (first["recent_noshows_count"] == 0).all()
    assert (first["bookings_last_30d"] == 0).all()
    assert (first["reliability_score"] == 100.0).all()
    later = frame.drop(first.index)
    assert (later["past_attendance_ratio"] != 0.85).any(), "history does change the features"


def test_the_features_are_computed_by_features_from_history_at_the_request_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, Any], list[dict[str, Any]], float, datetime]] = []
    real = simulate_module.features_from_history

    def spy(booking: dict[str, Any], history: list[dict[str, Any]], reliability: float, now: datetime, *rest: Any):
        calls.append((booking, list(history), reliability, now))
        return real(booking, history, reliability, now, *rest)

    monkeypatch.setattr(simulate_module, "features_from_history", spy)
    data = simulate(25, 3)
    assert len(calls) == len(data)
    for booking, history, _, now in calls:
        assert now == booking["requestedAt"], "scored at the moment the booking was requested"
        for row in history:
            assert row["requestedAt"] <= now, "only bookings requested earlier are visible"
            assert row["status"] in {"completed", "no_show", "booked"}
            assert row["startTime"] < now or row["status"] == "booked", "an unfinished booking has no outcome"


def test_no_row_uses_a_later_booking() -> None:
    bookings = generate_bookings(150, 11)
    features = build_features(bookings)
    order = bookings["requested_at"].sort_values()
    cutoff = order.iloc[len(order) // 2]
    early = bookings["requested_at"] <= cutoff

    # (a) throwing away every booking requested after the cutoff changes nothing before it
    truncated = build_features(bookings[early])
    pd.testing.assert_frame_equal(truncated, features.loc[truncated.index])

    # (b) flipping the outcome of every booking that starts after the cutoff (including the row's own)
    # changes no feature of a row requested before it: an outcome is unknown until the booking has started
    flipped = bookings.copy()
    late_start = flipped["start"] >= cutoff
    flipped.loc[late_start, "no_show"] = 1 - flipped.loc[late_start, "no_show"]
    after_flip = build_features(flipped)
    pd.testing.assert_frame_equal(after_flip.loc[early, FEATURE_ORDER], features.loc[early, FEATURE_ORDER])
    # ... and the flip is visible to the rows that may legitimately see those outcomes
    changed = (after_flip.loc[~early, FEATURE_ORDER] != features.loc[~early, FEATURE_ORDER]).any(axis=1)
    assert changed.any(), "the perturbation must be able to show up, otherwise the check proves nothing"


def _timeline() -> pd.DataFrame:
    base = pd.Timestamp.now(tz="UTC").floor("D")

    def booking(student: int, req_h: float, start_h: float, hours: float, kind: str, no_show: int) -> dict[str, Any]:
        return {
            "student_id": student,
            "requested_at": base + pd.Timedelta(hours=req_h),
            "start": base + pd.Timedelta(hours=start_h),
            "end": base + pd.Timedelta(hours=start_h + hours),
            "resource_type": kind,
            "no_show": no_show,
        }

    rows = [
        # student 1 (hand-worked in the test below); the list is deliberately not in time order
        booking(1, 60, 96, 2, "seat", 1),   # b3
        booking(1, 0, 24, 2, "seat", 1),    # b0
        booking(7, 20, 22, 1, "seat", 1),   # another student's miss, requested between b1 and b2
        booking(1, 30, 72, 2, "seat", 0),   # b2
        booking(1, 12, 48, 1, "room", 0),   # b1
    ]
    return pd.DataFrame(rows)


def test_a_booking_sees_earlier_requests_but_only_the_outcomes_that_were_known() -> None:
    bookings = _timeline()
    out = build_features(bookings)
    cols = ["past_attendance_ratio", "consecutive_attended_count", "recent_noshows_count",
            "bookings_last_30d", "reliability_score"]

    def at(student: int, nth: int) -> list[Any]:
        sub = bookings[bookings["student_id"] == student].sort_values("requested_at")
        return out.loc[sub.index[nth], cols].tolist()

    # b0 (requested at hour 0): no history at all
    assert at(1, 0) == [0.85, 0, 0, 0, 100.0]
    # b1 (hour 12): b0 was requested but starts at hour 24, so it is only a request: no outcome yet
    assert at(1, 1) == [0.85, 0, 0, 1, 100.0]
    # b2 (hour 30): b0 is a known miss (it started at 24); b1 starts at 48, still only a request;
    # the other student's miss never counts. 0 of 1 attended, 1 recent miss, 2 bookings, 100 - 10
    assert at(1, 2) == [0.0, 0, 1, 2, 90.0]
    # b3 (hour 60): b0 missed, b1 attended (it started at 48), b2 (starts at 72) only a request.
    # ratio 1/2; one attended after the last miss; 3 bookings; 100 - 10 + 2
    assert at(1, 3) == [0.5, 1, 1, 3, 92.0]
    # student 7's only booking is a first booking
    assert at(7, 0) == [0.85, 0, 0, 0, 100.0]


def test_the_reliability_score_follows_start_100_plus_2_per_attended_minus_10_per_miss() -> None:
    base = pd.Timestamp.now(tz="UTC").floor("D")
    rows: list[dict[str, Any]] = []

    def stream(student: int, outcomes: list[int]) -> None:
        for k, no_show in enumerate(outcomes):
            start = base + pd.Timedelta(hours=24 * k + 3)
            rows.append({
                "student_id": student,
                "requested_at": base + pd.Timedelta(hours=24 * k + 1),
                "start": start,
                "end": start + pd.Timedelta(hours=1),
                "resource_type": "seat",
                "no_show": no_show,
            })

    stream(1, [1] * 12 + [0, 0])     # twelve misses in a row, then two attended
    stream(2, [1] + [0] * 7)         # one miss, then seven attended
    bookings = pd.DataFrame(rows)
    out = build_features(bookings)

    def scores(student: int) -> list[float]:
        return out[bookings["student_id"] == student]["reliability_score"].tolist()

    # before each booking: 100, then -10 per miss, floored at 0 (10 misses reach 0; more stay 0),
    # then +2 for the attended booking that precedes the last row
    assert scores(1) == [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, 0.0, 0.0, 2.0]
    # 100, miss -> 90, then +2 each: 92, 94, 96, 98, 100, and the cap holds at 100
    assert scores(2) == [100.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 100.0]


def test_the_raw_bookings_span_about_120_days_in_utc_and_are_requested_before_they_start() -> None:
    bookings = generate_bookings(40, 9)
    window = bookings["start"].max() - bookings["start"].min()
    assert timedelta(days=100) <= window.to_pytimedelta() <= timedelta(days=125)
    assert str(bookings["start"].dt.tz) == "UTC" and str(bookings["requested_at"].dt.tz) == "UTC"
    assert ((bookings["start"] - bookings["requested_at"]) >= pd.Timedelta(minutes=5)).all()
    assert ((bookings["end"] - bookings["start"]) >= pd.Timedelta(minutes=60)).all()
    # sorted by student, then by the moment each booking was requested
    ordered = bookings.sort_values(["student_id", "requested_at"], kind="stable")
    assert ordered.index.equals(bookings.index)
