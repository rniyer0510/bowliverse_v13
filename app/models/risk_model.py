from pydantic import BaseModel, Field
from typing import Optional


class RiskModel(BaseModel):
    overall: str = "Low"

    # Existing / future extensible fields
    elbow: Optional[str] = None
    shoulder: Optional[str] = None
    knee: Optional[str] = None
