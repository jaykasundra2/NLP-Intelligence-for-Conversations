from fastapi import FastAPI
from api.chat_summary import chat_summary

app = FastAPI(
    openapi_url="/api/v1/chat_summary/openapi.json",
    docs_url="/api/v1/chat_summary/docs",
)

app.include_router(
    chat_summary, prefix="/api/v1/chat_summary", tags=["chat_summary"]
)