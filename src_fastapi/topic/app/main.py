from fastapi import FastAPI
from api.topic import topic

app = FastAPI(
    openapi_url="/api/v1/topic/openapi.json",
    docs_url="/api/v1/topic/docs",
)

app.include_router(
    topic, prefix="/api/v1/topic", tags=["topic"]
)