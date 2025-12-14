# app/pipeline/shoulder_alignment.py

import numpy as np
from app.models.context import Context
from app.utils.logger import log


# MediaPipe landmark indices (image-space, stable)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def run(ctx: Context) -> Context:
    """
    Shoulder alignment using image-space landmarks at FFC.
    Camera-relative (side-on capture assumed).

    Output:
        ctx.biomech.shoulder = {
            "angle_deg": float,
            "zone": "SIDE_ON" | "SEMI_OPEN" | "FRONT_ON",
            "frame": int
        }
    """

    events = ctx.events
    pose_frames = ctx.pose.frames

    # Default: feature absent
    ctx.biomech.shoulder = None

    if not events.ffc:
        log("[DEBUG] ShoulderAlign: FFC not available, skipping")
        return ctx

    f_ffc = events.ffc.frame
    if not (0 <= f_ffc < len(pose_frames)):
        log("[DEBUG] ShoulderAlign: Invalid FFC frame index")
        return ctx

    pf = pose_frames[f_ffc]
    if pf.landmarks is None:
        log("[DEBUG] ShoulderAlign: Missing landmarks at FFC")
        return ctx

    try:
        lm = pf.landmarks

        LS = np.array([lm[LEFT_SHOULDER]["x"], lm[LEFT_SHOULDER]["y"]])
        RS = np.array([lm[RIGHT_SHOULDER]["x"], lm[RIGHT_SHOULDER]["y"]])

        dx = RS[0] - LS[0]
        dy = RS[1] - LS[1]

        angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        if angle_deg < 20:
            zone = "SIDE_ON"
        elif angle_deg <= 45:
            zone = "SEMI_OPEN"
        else:
            zone = "FRONT_ON"

        ctx.biomech.shoulder = {
            "angle_deg": round(float(angle_deg), 2),
            "zone": zone,
            "frame": f_ffc,
        }

        log(
            f"[DEBUG] ShoulderAlign @FFC={f_ffc}: "
            f"angle={angle_deg:.2f}, zone={zone}"
        )

    except Exception as e:
        log(f"[WARN] ShoulderAlign failed: {e}")
        ctx.biomech.shoulder = None

    return ctx

