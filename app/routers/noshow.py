from fastapi import APIRouter

from app.schemas.noshow import NoShowRequest, NoShowResponse

router = APIRouter(prefix="/predict")


@router.post("/no-show", response_model=NoShowResponse)
def predict_no_show(payload: NoShowRequest) -> NoShowResponse:
    # Placeholder until the XGBoost model is trained in Phase 3. The contract is
    # real so the Node backend can be wired against it now.
    return NoShowResponse(predicted_probability=0.0, model_version="stub")
