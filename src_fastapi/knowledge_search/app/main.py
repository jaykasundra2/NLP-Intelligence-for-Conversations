from fastapi import FastAPI
from api.knowledge_search import knowledge_search

app = FastAPI(
    openapi_url="/api/v1/knowledge_search/openapi.json",
    docs_url="/api/v1/knowledge_search/docs",
)

app.include_router(
    knowledge_search, prefix="/api/v1/knowledge_search", tags=["knowledge_search"]
)