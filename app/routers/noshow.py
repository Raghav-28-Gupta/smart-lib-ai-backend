import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.ml.noshow.features import NoShowModel, from_request
from app.ml.noshow.model import NoShowModel as TrainedNoShowModel
from app.schemas.noshow import NoShowRequest, NoShowResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict")


@lru_cache(maxsize=1)
def load_committed_model() -> NoShowModel | None:
    """Load the newest artifact committed under app/ml/artifacts, once per process.

    Returns None when the directory holds no artifact or the artifact is corrupt or unreadable (the
    endpoint then answers 503 `model_unavailable` rather than crashing). The warning names only the
    class of the error: never a path or the exception text. The result, None included, is cached, so
    a broken artifact is not re-read on every request; fix it and restart.
    """
    try:
        return TrainedNoShowModel.load()
    except Exception as error:
        cause = error.__cause__ or error
        logger.warning(
            "no-show model not loaded (%s); /predict/no-show will answer 503", type(cause).__name__
        )
        return None


def get_model() -> NoShowModel | None:
    """The loaded no-show model, or None. Tests replace this dependency through app.dependency_overrides."""
    return load_committed_model()


@router.post("/no-show", response_model=NoShowResponse)
def predict_no_show(
    payload: NoShowRequest,
    model: NoShowModel | None = Depends(get_model),
) -> NoShowResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="model_unavailable")
    probability, penalty_factor, model_version = model.predict(from_request(payload))
    return NoShowResponse(
        predicted_probability=probability,
        penalty_factor=penalty_factor,
        model_version=model_version,
    )
