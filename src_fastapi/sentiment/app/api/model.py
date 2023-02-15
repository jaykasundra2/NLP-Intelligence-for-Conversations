# Importing libraries
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# Model for receiving input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


# Model for sentiment analysis service response
class SentimentResponse(BaseModel):
    response: dict
