# Importing libraries
from pydantic import BaseModel
from typing import Optional

# Model for receiving input
class Input(BaseModel):
    model: str
    text: str
    topic_list: str

# Model for service response
class TopicResponse(BaseModel):
    response: dict