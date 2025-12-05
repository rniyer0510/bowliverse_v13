from pydantic import BaseModel
from typing import Literal

class InputModel(BaseModel):
    file_path: str
    hand: Literal["R", "L"]
    bowler_type: str
