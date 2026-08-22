from pydantic import BaseModel, Field


class NoShowRequest(BaseModel):
    user_id: str
    resource_booking_id: str


class NoShowResponse(BaseModel):
    predicted_probability: float = Field(ge=0.0, le=1.0)
    model_version: str
