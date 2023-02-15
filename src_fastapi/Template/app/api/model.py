# Importing libraries
from pydantic import BaseModel
from typing import Optional


# Model for receiving input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None
