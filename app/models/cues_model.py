from pydantic import BaseModel, Field
from typing import List

class CuesModel(BaseModel):
    list: List[str] = Field(default_factory=list)
