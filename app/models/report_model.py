from pydantic import BaseModel, Field
from typing import List, Optional


class PostureRow(BaseModel):
    phase: str
    parameter: str
    classification: Optional[str] = None
    value: Optional[str] = None
    risk: str = "Low"


class ReportSummary(BaseModel):
    legality: str
    risk: str
    action_type: str
    confidence_pct: int


class ReportInterpretation(BaseModel):
    summary_text: str
    notes: List[str] = Field(default_factory=list)


class ReportModel(BaseModel):
    schema_id: str = "bowliverse.v13"
    version: str = "13.0.0"

    posture_table: List[PostureRow] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    summary: Optional[ReportSummary] = None
    interpretation: Optional[ReportInterpretation] = None
