from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ml.noshow.features import MAX_PENALTY_CAP

# The counts have no meaningful upper bound (the backend caps them at 20, 10 and 30), but they must
# stay small enough to become a float: a 400-digit integer would otherwise be a 500.
MAX_COUNT = 1_000_000


class NoShowRequest(BaseModel):
    """The ten features of one booking (spec section 6). Field order is FEATURE_ORDER.

    The Node backend builds these from its database; this service is stateless. Unknown
    fields are rejected so a drifting client fails loudly instead of being half understood,
    and floats must be finite (NaN and infinity are never valid features).
    """

    model_config = ConfigDict(extra="forbid")

    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="Monday = 0")
    duration_minutes: int = Field(ge=1, le=1440)
    lead_time_hours: float = Field(ge=0.0, allow_inf_nan=False)
    resource_type: Literal["seat", "room"]
    past_attendance_ratio: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    consecutive_attended_count: int = Field(ge=0, le=MAX_COUNT)
    recent_noshows_count: int = Field(ge=0, le=MAX_COUNT)
    reliability_score: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    bookings_last_30d: int = Field(ge=0, le=MAX_COUNT)


class NoShowResponse(BaseModel):
    predicted_probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    penalty_factor: float = Field(ge=0.0, le=MAX_PENALTY_CAP, allow_inf_nan=False)
    model_version: str
