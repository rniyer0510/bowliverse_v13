# app/pipeline/shoulder_alignment.py

import numpy as np
from app.models.context import Context
from app.utils.logger import log


# MediaPipe landmark indices (image-space, stable)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def run(ctx: Context) -> Context:
    """
    Shoulder alignment using image-space landmarks.

    Interpretation (side-on camera assumed):
    - Shoulder line parallel to pitch  → SIDE_ON
    - Shoulder line perpendicular      → FRONT_ON
    - Transitional / misaligned        → MIXED

    Output:
        ctx.biomech.shoulder = {
            "angle_deg": float,
            "zone": "SIDE_ON" | "MIXED" | "FRONT_ON",
            "frame": int
        }
    """

    events = ctx.events
    pose_frames = ctx.pose.frames

    # Default: feature absent
    ctx.biomech.shoulder = None

    # Prefer UAH for action classification
    if not events.uah:
        log("[DEBUG] ShoulderAlign: UAH not available, skipping")
        return ctx

    frame = events.uah.frame
    if not (0 <= frame < len(pose_frames)):
        log("[DEBUG] ShoulderAlign: Invalid UAH frame index")
        return ctx

    pf = pose_frames[frame]
    if pf.landmarks is None:
        log("[DEBUG] ShoulderAlign: Missing landmarks at UAH")
        return ctx

    try:
        lm = pf.landmarks

        LS = np.array([lm[LEFT_SHOULDER]["x"], lm[LEFT_SHOULDER]["y"]])
        RS = np.array([lm[RIGHT_SHOULDER]["x"], lm[RIGHT_SHOULDER]["y"]])

        dx = RS[0] - LS[0]
        dy = RS[1] - LS[1]

        # Angle of shoulder line in image plane
        angle_deg = abs(np.degrees(np.arctan2(dy, dx)))

        # Normalize to [0–90]
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        # -------------------------------------------------
        # Biomechanically correct zones (range-based)
        # -------------------------------------------------
        # ~0°  → parallel to pitch → SIDE_ON
        # ~90° → perpendicular     → FRONT_ON
        if angle_deg <= 25:
            zone = "SIDE_ON"
        elif angle_deg >= 65:
            zone = "FRONT_ON"
        else:
            zone = "MIXED"

        ctx.biomech.shoulder = {
            "angle_deg": round(float(angle_deg), 2),
            "zone": zone,
            "frame": int(frame),
        }

        log(
            f"[DEBUG] ShoulderAlign @UAH={frame}: "
            f"angle={angle_deg:.2f}, zone={zone}"
        )

    except Exception as e:
        log(f"[WARN] ShoulderAlign failed: {e}")
        ctx.biomech.shoulder = None

    return ctx

