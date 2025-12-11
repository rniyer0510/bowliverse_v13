from pydantic import BaseModel
from typing import Optional

class EventFrame(BaseModel):
    frame: int
    conf: float

class EventsModel(BaseModel):
    release: Optional[EventFrame] = None
    uah: Optional[EventFrame] = None
    ffc: Optional[EventFrame] = None
    bfc: Optional[EventFrame] = None
    error: Optional[str] = None
