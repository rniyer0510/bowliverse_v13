from pydantic import BaseModel
from typing import Optional


# ----------------------------
# Elbow biomechanics
# ----------------------------
class BiomechElbowModel(BaseModel):
    uah_angle: Optional[float] = None
    release_angle: Optional[float] = None

    peak_extension_angle_deg: Optional[float] = None
    peak_extension_frame: Optional[int] = None

    extension_deg: Optional[float] = None
    extension_raw_deg: Optional[float] = None
    extension_error_margin_deg: Optional[float] = None
    extension_note: Optional[str] = None


# ----------------------------
# Release height
# ----------------------------
class ReleaseHeightModel(BaseModel):
    norm_height: Optional[float] = None
    wrist_y: Optional[float] = None


# ----------------------------
# Backfoot @ BFC primitive
# ----------------------------
class BackFootModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


# ----------------------------
# Hip @ FFC primitive
# ----------------------------
class HipModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


# ----------------------------
# Shoulder @ FFC primitive
# ----------------------------
class ShoulderModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None

# ----------------------------
# Shoulder–Hip separation @ FFC
# ----------------------------
class ShoulderHipModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


# ----------------------------
# Aggregate biomech model
# ----------------------------
class BiomechModel(BaseModel):
    elbow: Optional[BiomechElbowModel] = None
    release_height: Optional[ReleaseHeightModel] = None

    # Alignment / posture primitives (matrix-ready)
    backfoot: Optional[BackFootModel] = None
    hip: Optional[HipModel] = None
    shoulder: Optional[ShoulderModel] = None
    shoulder_hip: Optional[ShoulderHipModel] = None

    elbow_conf: Optional[float] = None
    release_height_conf: Optional[float] = None
    angle_plane: Optional[dict] = None

    error: Optional[str] = None

