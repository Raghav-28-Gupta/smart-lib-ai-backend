"""Contract tests for POST /predict/no-show (spec 2026-11-24-no-show-handling, section 6).

The service is stateless: the Node backend sends ten features and gets a probability, a bounded
penalty factor and a model version back. A fake model is injected through the get_model
dependency, so nothing here needs a trained artifact.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.ml.noshow.features import FEATURE_ORDER, MAX_PENALTY_CAP
from app.schemas.noshow import NoShowRequest, NoShowResponse

URL = "/predict/no-show"

# The order the spec gives the features in; the model is trained and queried in this order.
EXPECTED_FEATURE_ORDER = [
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

# One value just outside each bound (and a few values of the wrong kind), per field.
INVALID_VALUES = [
    ("hour_of_day", -1),
    ("hour_of_day", 24),
    ("hour_of_day", 10.5),
    ("day_of_week", -1),
    ("day_of_week", 7),
    ("duration_minutes", 0),
    ("duration_minutes", 1441),
    ("duration_minutes", 90.5),
    ("lead_time_hours", -0.5),
    ("past_attendance_ratio", -0.01),
    ("past_attendance_ratio", 1.01),
    ("consecutive_attended_count", -1),
    ("recent_noshows_count", -1),
    ("reliability_score", -0.5),
    ("reliability_score", 100.5),
    ("bookings_last_30d", -1),
    ("hour_of_day", "morning"),
    ("lead_time_hours", None),
]

# Exactly on each bound is valid; the counts and the lead time have no upper bound.
BOUNDARY_VALUES = [
    ("hour_of_day", 0),
    ("hour_of_day", 23),
    ("day_of_week", 0),
    ("day_of_week", 6),
    ("duration_minutes", 1),
    ("duration_minutes", 1440),
    ("lead_time_hours", 0),
    ("lead_time_hours", 24.0 * 365),
    ("resource_type", "seat"),
    ("resource_type", "room"),
    ("past_attendance_ratio", 0),
    ("past_attendance_ratio", 1),
    ("consecutive_attended_count", 0),
    ("consecutive_attended_count", 500),
    ("recent_noshows_count", 0),
    ("recent_noshows_count", 500),
    ("reliability_score", 0),
    ("reliability_score", 100),
    ("bookings_last_30d", 0),
    ("bookings_last_30d", 500),
]

NON_FINITE_FLOAT_FIELDS = ["lead_time_hours", "past_attendance_ratio", "reliability_score"]


def _post_raw(client: Any, payload: dict[str, Any]) -> Any:
    """Post with Python's json.dumps spelling of NaN/Infinity, which httpx's json= refuses to send."""
    return client.post(
        URL,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


# --- the happy path -----------------------------------------------------------------------------


def test_valid_request_returns_probability_penalty_and_version(client_with_model, valid_payload):
    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "predicted_probability": 0.31,
        "penalty_factor": 0.155,
        "model_version": "noshow-test",
    }


def test_model_receives_the_ten_features_in_order(client_with_model, fake_model, valid_payload):
    client_with_model.post(URL, json=valid_payload)

    assert len(fake_model.calls) == 1
    assert list(fake_model.calls[0]) == EXPECTED_FEATURE_ORDER
    assert fake_model.calls[0] == valid_payload


@pytest.mark.parametrize(("field", "value"), BOUNDARY_VALUES)
def test_values_exactly_on_a_bound_are_accepted(client_with_model, valid_payload, field, value):
    valid_payload[field] = value

    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 200, response.text


# --- validation: 422 ----------------------------------------------------------------------------


@pytest.mark.parametrize("field", EXPECTED_FEATURE_ORDER)
def test_each_missing_field_is_a_422(client_with_model, valid_payload, field):
    del valid_payload[field]

    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert [(error["type"], error["loc"]) for error in errors] == [("missing", ["body", field])]


@pytest.mark.parametrize(("field", "value"), INVALID_VALUES)
def test_each_out_of_range_value_is_a_422(client_with_model, fake_model, valid_payload, field, value):
    valid_payload[field] = value

    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
    assert fake_model.calls == []


@pytest.mark.parametrize("value", ["desk", "Seat", "ROOM", "", None, 1])
def test_unknown_resource_type_is_a_422(client_with_model, valid_payload, value):
    valid_payload["resource_type"] = value

    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "resource_type"]


@pytest.mark.parametrize("extra", [{"user_id": "u1"}, {"resource_booking_id": "b1"}, {"surprise": 1}])
def test_an_extra_field_is_a_422(client_with_model, valid_payload, extra):
    valid_payload.update(extra)

    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert [(error["type"], error["loc"]) for error in errors] == [
        ("extra_forbidden", ["body", name]) for name in extra
    ]


def test_the_old_stub_request_shape_is_a_422(client_with_model):
    response = client_with_model.post(URL, json={"user_id": "u1", "resource_booking_id": "b1"})

    assert response.status_code == 422


@pytest.mark.parametrize("field", NON_FINITE_FLOAT_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_a_422(client_with_model, fake_model, valid_payload, field, value):
    valid_payload[field] = value

    response = _post_raw(client_with_model, valid_payload)

    assert response.status_code == 422
    assert fake_model.calls == []


COUNT_FIELDS = ["consecutive_attended_count", "recent_noshows_count", "bookings_last_30d"]
NUMBER_FIELDS = [name for name in EXPECTED_FEATURE_ORDER if name != "resource_type"]


@pytest.mark.parametrize("field", COUNT_FIELDS)
def test_a_count_may_be_a_million_but_not_a_million_and_one(client_with_model, valid_payload, field):
    valid_payload[field] = 1_000_000
    assert client_with_model.post(URL, json=valid_payload).status_code == 200

    valid_payload[field] = 1_000_001
    response = client_with_model.post(URL, json=valid_payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize("field", NUMBER_FIELDS)
def test_a_400_digit_integer_is_a_422_not_a_500(client_with_model, fake_model, valid_payload, field):
    valid_payload[field] = 10**400

    response = _post_raw(client_with_model, valid_payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
    assert fake_model.calls == []


@pytest.mark.parametrize("field", NUMBER_FIELDS)
def test_a_number_too_large_for_a_double_is_a_422(client_with_model, fake_model, valid_payload, field):
    """JSON `1e400` parses to infinity in Python."""
    valid_payload[field] = "TOO_BIG"
    body = json.dumps(valid_payload).replace('"TOO_BIG"', "1e400").encode()

    response = client_with_model.post(URL, content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
    assert fake_model.calls == []


def test_a_body_that_is_not_json_is_a_422(client_with_model):
    response = client_with_model.post(URL, content=b"not json", headers={"content-type": "application/json"})

    assert response.status_code == 422


# --- no model: 503 ------------------------------------------------------------------------------


def test_503_model_unavailable_without_a_model(client, valid_payload):
    response = client.post(URL, json=valid_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}


def test_invalid_input_is_a_422_even_without_a_model(client, valid_payload):
    valid_payload["hour_of_day"] = 99

    response = client.post(URL, json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", NUMBER_FIELDS)
def test_a_400_digit_integer_is_a_422_with_the_real_model_too(committed_client, valid_payload, field):
    valid_payload[field] = 10**400

    response = _post_raw(committed_client, valid_payload)

    assert response.status_code == 422


# --- the schema and the feature list ------------------------------------------------------------


def test_feature_order_is_the_ten_features_in_the_specs_order():
    assert FEATURE_ORDER == EXPECTED_FEATURE_ORDER


def test_feature_order_equals_the_requests_fields():
    assert FEATURE_ORDER == list(NoShowRequest.model_fields)


def test_max_penalty_cap_is_a_fifth():
    assert MAX_PENALTY_CAP == 0.2


@pytest.mark.parametrize(
    ("probability", "penalty"),
    [(0.0, 0.0), (1.0, 0.2), (0.31, 0.155)],
)
def test_response_accepts_values_inside_the_bounds(probability, penalty):
    response = NoShowResponse(predicted_probability=probability, penalty_factor=penalty, model_version="v")

    assert response.penalty_factor == penalty


@pytest.mark.parametrize(
    ("probability", "penalty"),
    [
        (-0.01, 0.1),
        (1.01, 0.1),
        (float("nan"), 0.1),
        (0.5, -0.01),
        (0.5, 0.2001),
        (0.5, float("nan")),
        (0.5, float("inf")),
    ],
)
def test_response_rejects_values_outside_the_bounds(probability, penalty):
    with pytest.raises(ValidationError):
        NoShowResponse(predicted_probability=probability, penalty_factor=penalty, model_version="v")


# --- the rest of the service --------------------------------------------------------------------


def test_health_still_answers(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai-backend"}
