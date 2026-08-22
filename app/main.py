from fastapi import FastAPI

from app.core.config import settings
from app.routers import health, noshow, recommend

app = FastAPI(title=settings.app_name)

app.include_router(health.router)
app.include_router(noshow.router)
app.include_router(recommend.router)
