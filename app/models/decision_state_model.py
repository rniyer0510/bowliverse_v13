from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class DecisionStateModel(BaseModel):
    """
    Holds interpretation-level decisions derived from biomechanics.
    JSON-first: action_matrix is a dict at runtime.
    """
    action_matrix: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Optional fields (safe placeholders)
    legality: Optional[str] = None
    confidence_pct: Optional[float] = None
