from pydantic import BaseModel, Field
from typing import List

class ReportModel(BaseModel):
    schema_id: str = "bowliverse.v13"
    version: str = "13.0.0"
    warnings: List[str] = Field(default_factory=list)
