from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PoseFrame(BaseModel):
    frame_index: int
    # Each landmark: {"x": float, "y": float, "z": float, "vis": float}
    landmarks: List[Dict[str, float]]
    confidence: float

class PoseModel(BaseModel):
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    duration_sec: Optional[float] = None
    frames: List[PoseFrame] = Field(default_factory=list)
    error: Optional[str] = None
