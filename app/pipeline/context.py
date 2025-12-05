from pydantic import BaseModel
from app.models.input_model import InputModel
from app.models.pose_model import PoseModel
from app.models.events_model import EventsModel
from app.models.biomech_model import BiomechModel
from app.models.cues_model import CuesModel
from app.models.risk_model import RiskModel
from app.models.report_model import ReportModel

class Context(BaseModel):
    input: InputModel
    pose: PoseModel = PoseModel()
    events: EventsModel = EventsModel()
    biomech: BiomechModel = BiomechModel()
    cues: CuesModel = CuesModel()
    risk: RiskModel = RiskModel()
    report: ReportModel = ReportModel()
