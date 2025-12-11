from pydantic import BaseModel
from typing import Optional


class BiomechElbowModel(BaseModel):
    uah_angle: Optional[float] = None
    release_angle: Optional[float] = None

    peak_extension_angle_deg: Optional[float] = None
    peak_extension_frame: Optional[int] = None

    extension_deg: Optional[float] = None
    extension_raw_deg: Optional[float] = None
    extension_error_margin_deg: Optional[float] = None
    extension_note: Optional[str] = None


class ReleaseHeightModel(BaseModel):
    norm_height: Optional[float] = None
    wrist_y: Optional[float] = None


class BiomechModel(BaseModel):
    elbow: Optional[BiomechElbowModel] = None
    release_height: Optional[ReleaseHeightModel] = None

    elbow_conf: Optional[float] = None
    release_height_conf: Optional[float] = None
    angle_plane: Optional[dict] = None

    error: Optional[str] = None

