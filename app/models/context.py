from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from app.models.input_model import InputModel
from app.models.video_model import VideoModel
from app.models.pose_model import PoseModel
from app.models.events_model import EventsModel
from app.models.biomech_model import BiomechModel
from app.models.risk_model import RiskModel
from app.models.cues_model import CuesModel
from app.models.decision_state_model import DecisionStateModel


class Context(BaseModel):
    """
    Canonical analysis context passed through the pipeline.

    NOTE:
    - Core pipeline stages (biomech/decision/report) are JSON-first.
    - We relax typing at the boundary to match runtime dict outputs and avoid
      noisy Pydantic serialization warnings.
    """

    # -------------------------
    # Inputs & raw data
    # -------------------------
    input: InputModel
    video: VideoModel = Field(default_factory=VideoModel)
    pose: PoseModel = Field(default_factory=PoseModel)

    # -------------------------
    # Derived stages
    # -------------------------
    events: EventsModel = Field(default_factory=EventsModel)
    biomech: BiomechModel = Field(default_factory=BiomechModel)

    # -------------------------
    # Interpretation layers
    # -------------------------
    decision: DecisionStateModel = Field(default_factory=DecisionStateModel)
    risk: RiskModel = Field(default_factory=RiskModel)
    cues: CuesModel = Field(default_factory=CuesModel)

    # -------------------------
    # Reporting (JSON-first)
    # -------------------------
    report: Optional[Dict[str, Any]] = Field(default_factory=dict)
