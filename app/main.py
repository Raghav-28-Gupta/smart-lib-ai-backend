from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.routers import health, noshow, recommend


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the committed no-show model at startup, so the first prediction is not the slow one."""
    noshow.load_committed_model()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with the type, location and message of each problem, and nothing of the input.

    FastAPI's default body echoes each rejected value back. A NaN or infinity there cannot be
    written as JSON, so rejecting one would turn a 422 into a 500.
    """
    detail = [
        {"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


app.include_router(health.router)
app.include_router(noshow.router)
app.include_router(recommend.router)
