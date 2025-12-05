from pydantic import BaseModel, Field
from typing import List, Optional, Any

class VideoModel(BaseModel):
    # Raw OpenCV frames - internal only, never exposed in JSON
    frames: List[Any] = Field(default_factory=list, exclude=True)

    frame_count: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    error: Optional[str] = None
