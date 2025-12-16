# app/pipeline/shoulder_hip_separation.py
"""
ActionLab / Bowliverse v13.9 — SHOULDER–HIP SEPARATION (FFC-ANCHORED)

What this computes:
- Relative rotation between shoulders and hips at Front Foot Contact (FFC)
- Uses ground-plane projection (X–Z) for camera robustness
- Positive separation indicates shoulders rotated more than hips (good for pace)

Design rules:
- Strictly anchored at FFC
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
    Computes shoulder–hip separation at FFC and stores result in ctx.biomech.shoulder_hip

    Output structure:
    {
        "separation_deg": float,
        "zone": "LOW" | "MODERATE" | "HIGH",
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
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    try:
        LS = lm[LEFT_SHOULDER]
        RS = lm[RIGHT_SHOULDER]
        LH = lm[LEFT_HIP]
        RH = lm[RIGHT_HIP]

        LS_pt = np.array([LS["x"], LS["y"], LS["z"]], float)
        RS_pt = np.array([RS["x"], RS["y"], RS["z"]], float)
        LH_pt = np.array([LH["x"], LH["y"], LH["z"]], float)
        RH_pt = np.array([RH["x"], RH["y"], RH["z"]], float)
    except Exception:
        return

    # -------------------------------------------------
    # Ground-plane vectors (X–Z)
    # -------------------------------------------------
    shoulder_vec = RS_pt - LS_pt
    hip_vec = RH_pt - LH_pt

    shoulder_xz = np.array([shoulder_vec[0], shoulder_vec[2]], float)
    hip_xz = np.array([hip_vec[0], hip_vec[2]], float)

    # -------------------------------------------------
    # Compute angles relative to neutral axis
    # -------------------------------------------------
    ref_axis = np.array([1.0, 0.0], float)

    sh_ang = _angle_2d(shoulder_xz, ref_axis)
    hip_ang = _angle_2d(hip_xz, ref_axis)

    if sh_ang is None or hip_ang is None:
        return

    # -------------------------------------------------
    # Handedness normalization
    # -------------------------------------------------
    if ctx.input.hand == "L":
        sh_ang = 180.0 - sh_ang
        hip_ang = 180.0 - hip_ang

    # Normalize both to [0, 90]
    sh_ang = abs(sh_ang)
    hip_ang = abs(hip_ang)

    if sh_ang > 90.0:
        sh_ang = 180.0 - sh_ang
    if hip_ang > 90.0:
        hip_ang = 180.0 - hip_ang

    # -------------------------------------------------
    # Separation (shoulders lead hips)
    # -------------------------------------------------
    separation = sh_ang - hip_ang

    # Clamp for safety
    separation = max(-90.0, min(90.0, separation))

    # -------------------------------------------------
    # Zone classification (tunable)
    # -------------------------------------------------
    abs_sep = abs(separation)

    if abs_sep < 10.0:
        zone = "LOW"
    elif abs_sep < 25.0:
        zone = "MODERATE"
    else:
        zone = "HIGH"

    # -------------------------------------------------
    # Store result
    # -------------------------------------------------
    ctx.biomech.shoulder_hip = {
        "separation_deg": round(float(separation), 2),
        "zone": zone,
        "confidence": round(float(conf), 2),
        "frame": int(frame),
    }

