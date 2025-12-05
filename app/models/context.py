from pydantic import BaseModel, Field
from app.models.input_model import InputModel
from app.models.video_model import VideoModel
from app.models.pose_model import PoseModel
from app.models.events_model import EventsModel
from app.models.biomech_model import BiomechModel
from app.models.risk_model import RiskModel
from app.models.cues_model import CuesModel
from app.models.report_model import ReportModel

class Context(BaseModel):
    input: InputModel
    video: VideoModel = Field(default_factory=VideoModel)
    pose: PoseModel = Field(default_factory=PoseModel)
    events: EventsModel = Field(default_factory=EventsModel)
    biomech: BiomechModel = Field(default_factory=BiomechModel)
    risk: RiskModel = Field(default_factory=RiskModel)
    cues: CuesModel = Field(default_factory=CuesModel)
    report: ReportModel = Field(default_factory=ReportModel)
