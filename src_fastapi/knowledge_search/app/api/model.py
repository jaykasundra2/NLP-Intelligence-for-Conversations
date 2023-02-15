# Importing libraries
from pydantic import BaseModel
from typing import Optional

# Model for receiving input
class Input(BaseModel):
    model: str
    text: str
    knowledge_articles : dict
    query: str

# Model for service response
class KnowledgeSearchResponse(BaseModel):
    response: dict