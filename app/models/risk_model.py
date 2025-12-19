from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RiskModel(BaseModel):
    """
    Top-level Risk model (Option A).

    - overall: aggregated risk level (Low / Medium / High)
    - breakdown: list of detected risk components
    - confidence: structural confidence of aggregation
    """

    overall: str = "Low"
    breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0

    # Reserved / future extensibility
    elbow: Optional[str] = None
    shoulder: Optional[str] = None
    knee: Optional[str] = None

