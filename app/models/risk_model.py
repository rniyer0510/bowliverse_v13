from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class RiskModel(BaseModel):
    score: Optional[float] = None
    level: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
