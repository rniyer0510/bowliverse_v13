# app/pipeline/hip_alignment.py

import numpy as np
from app.models.context import Context
from app.utils.angles import angle


def run(ctx: Context) -> None:
    """
    Hip orientation at Front Foot Contact (FFC).
    Computes orientation of hip line relative to a pitch reference.

    Output: ctx.biomech.hip = {angle_deg, zone, confidence, frame}

    NOTE:
    - Uses existing angles.angle(a,b,c) for consistency.
    - Uses ctx.pose.frames as a LIST.
    - Pitch reference here is +Y placeholder; we will calibrate axis & thresholds later.
    """

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

    # MediaPipe hips
    LEFT_HIP = 23
    RIGHT_HIP = 24

    L = lm[LEFT_HIP]
    R = lm[RIGHT_HIP]

    L_pt = np.array([L["x"], L["y"], L["z"]], float)
    R_pt = np.array([R["x"], R["y"], R["z"]], float)

    center = (L_pt + R_pt) / 2.0
    pitch_pt = center + np.array([0.0, 1.0, 0.0])  # forward reference

    # Angle at center between hip-line direction and pitch
    # Using R as the hip-line direction anchor
    ang = float(angle(R_pt, center, pitch_pt))

    # Practical zones (tunable)
    if ang <= 30:
        zone = "SIDE_ON"
    elif ang <= 50:
        zone = "TRANSITIONAL"
    else:
        zone = "FRONT_ON"

    ctx.biomech.hip = {
        "angle_deg": round(ang, 2),
        "zone": zone,
        "confidence": round(float(conf), 2),
        "frame": int(frame),
    }
