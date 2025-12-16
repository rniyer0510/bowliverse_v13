# app/pipeline/backfoot_landing_angle.py

import numpy as np
from app.models.context import Context
from app.utils.logger import log


def _angle_2d(v):
    """Angle of vector in X–Z plane (degrees)."""
    x, z = v[0], v[2]
    return float(np.degrees(np.arctan2(z, x)))


def run(ctx: Context) -> None:
    """
    Compute Back Foot Contact (BFC) landing angle.
    Reverse search anchored at FFC.
    """

    if not ctx.events or not ctx.events.ffc:
        return

    pose_frames = ctx.pose.frames
    ffc_idx = ctx.events.ffc.frame

    if ffc_idx <= 1 or ffc_idx >= len(pose_frames):
        return

    # MediaPipe indices
    if ctx.input.hand == "R":
        ANKLE = 28      # right ankle
        HEEL = 30
        TOE = 32
    else:
        ANKLE = 27      # left ankle
        HEEL = 29
        TOE = 31

    ankle_y = []

    for i in range(ffc_idx):
        lm = pose_frames[i].landmarks
        if lm is None:
            ankle_y.append(None)
        else:
            ankle_y.append(float(lm[ANKLE]["y"]))

    # Normalize missing values
    ankle_y = [1.0 if v is None else v for v in ankle_y]

    # Reverse search for contact transition
    ground_thresh = np.percentile(ankle_y, 10)

    bfc_frame = None
    for i in range(ffc_idx - 1, 1, -1):
        if ankle_y[i] <= ground_thresh and ankle_y[i - 1] > ground_thresh:
            bfc_frame = i
            break

    if bfc_frame is None:
        log("[WARN] BFC landing not detected")
        return

    pf = pose_frames[bfc_frame]
    lm = pf.landmarks
    if lm is None:
        return

    heel = np.array([lm[HEEL]["x"], lm[HEEL]["y"], lm[HEEL]["z"]])
    toe = np.array([lm[TOE]["x"], lm[TOE]["y"], lm[TOE]["z"]])

    foot_vec = toe - heel
    angle = abs(_angle_2d(foot_vec))

    # Interpretation ranges
    if angle < 25:
        zone = "CLOSED"
    elif angle < 45:
        zone = "NEUTRAL"
    elif angle < 65:
        zone = "OPEN"
    else:
        zone = "VERY_OPEN"

    ctx.biomech.backfoot = {
        "landing_angle_deg": round(angle, 2),
        "zone": zone,
        "confidence": round(float(ctx.events.ffc.conf), 2),
        "frame": int(bfc_frame),
    }

    log(
        f"[DEBUG] BFC Landing @frame={bfc_frame}: "
        f"angle={angle:.2f}, zone={zone}"
    )

