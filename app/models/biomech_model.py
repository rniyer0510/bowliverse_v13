from pydantic import BaseModel
from typing import Optional, Dict, Any


# ----------------------------
# Elbow biomechanics (kept as documentation)
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
# Release height (documentation)
# ----------------------------
class ReleaseHeightModel(BaseModel):
    norm_height: Optional[float] = None
    wrist_y: Optional[float] = None


# ----------------------------
# Backfoot / Hip / Shoulder / ShoulderHip (documentation)
# ----------------------------
class BackFootModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


class HipModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


class ShoulderModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


class ShoulderHipModel(BaseModel):
    angle_deg: Optional[float] = None
    zone: Optional[str] = None
    confidence: Optional[float] = None
    frame: Optional[int] = None


# ----------------------------
# Aggregate biomech model
# NOTE: JSON-first fields to match runtime (dicts)
# ----------------------------
class BiomechModel(BaseModel):
    # Biomech computed blocks are produced as dicts at runtime
    elbow: Optional[Dict[str, Any]] = None
    release_height: Optional[Dict[str, Any]] = None

    # Alignment / posture primitives are dicts at runtime
    backfoot: Optional[Dict[str, Any]] = None
    hip: Optional[Dict[str, Any]] = None
    shoulder: Optional[Dict[str, Any]] = None
    shoulder_hip: Optional[Dict[str, Any]] = None

    # Confidence + misc blocks
    elbow_conf: Optional[float] = None
    release_height_conf: Optional[float] = None
    angle_plane: Optional[Dict[str, Any]] = None

    error: Optional[str] = None
