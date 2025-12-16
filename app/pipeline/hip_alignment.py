# app/pipeline/hip_alignment.py
"""
ActionLab / Bowliverse v13.9 — HIP ALIGNMENT (FFC-ANCHORED, CAMERA-ROBUST)

What this computes:
- Pelvis (hip-line) orientation at Front Foot Contact (FFC)
- Uses ground-plane projection (X–Z), ignores Y to avoid camera tilt issues
- Outputs angle + zone for action matrix and downstream logic

Design rules:
- Anchor strictly at FFC
- No pitch-direction assumptions
- Camera-robust under side-on / near side-on capture
- Graceful degradation if landmarks are missing
"""

import numpy as np
from app.models.context import Context


# ---------------------------------------------------------
# Utility: angle in degrees between 2D vectors
# ---------------------------------------------------------
def _angle_2d(v1, v2):
    """
    Returns angle in degrees between two 2D vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(ctx: Context) -> None:
    """
    Computes hip orientation at FFC and stores result in ctx.biomech.hip

    Output structure:
    {
        "angle_deg": float,
        "zone": "SIDE_ON" | "TRANSITIONAL" | "FRONT_ON",
        "confidence": float,
        "frame": int
    }
    """

    # -------------------------------------------------
    # Preconditions
    # -------------------------------------------------
    if not ctx.events or not ctx.events.ffc:
        return

    frame = ctx.events.ffc.frame
    conf = ctx.events.ffc.conf

    pose_frames = ctx.pose.frames
    if frame < 0 or frame >= len(pose_frames):
        return

    pf = pose_frames[frame]
    if pf.landmarks is None:
        return

    lm = pf.landmarks

    # MediaPipe landmark indices
    LEFT_HIP = 23
    RIGHT_HIP = 24

    try:
        L = lm[LEFT_HIP]
        R = lm[RIGHT_HIP]

        L_pt = np.array([L["x"], L["y"], L["z"]], float)
        R_pt = np.array([R["x"], R["y"], R["z"]], float)
    except Exception:
        return

    # -------------------------------------------------
    # Hip vector (projected to ground plane: X–Z)
    # -------------------------------------------------
    hip_vec = R_pt - L_pt
    hip_vec_xz = np.array([hip_vec[0], hip_vec[2]], float)

    # Reference axis: camera horizontal (X–Z forward)
    # We do NOT assume pitch direction; this is a neutral axis
    ref_axis = np.array([1.0, 0.0], float)

    ang = _angle_2d(hip_vec_xz, ref_axis)
    if ang is None:
        return

    # -------------------------------------------------
    # Handedness normalization
    # -------------------------------------------------
    # Left-handed bowlers mirror the pelvis orientation
    if ctx.input.hand == "L":
        ang = 180.0 - ang

    # Normalize to [0, 90]
    ang = abs(ang)
    if ang > 90.0:
        ang = 180.0 - ang

    # -------------------------------------------------
    # Zone classification (tunable, but stable)
    # -------------------------------------------------
    if ang <= 30.0:
        zone = "SIDE_ON"
    elif ang <= 55.0:
        zone = "TRANSITIONAL"
    else:
        zone = "FRONT_ON"

    # -------------------------------------------------
    # Store result
    # -------------------------------------------------
    ctx.biomech.hip = {
        "angle_deg": round(float(ang), 2),
        "zone": zone,
        "confidence": round(float(conf), 2),
        "frame": int(frame),
    }

