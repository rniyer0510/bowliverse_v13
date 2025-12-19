import numpy as np
from app.models.context import Context
from app.models.events_model import EventFrame
from app.utils.logger import log


def _angle_2d(v):
    """Angle of vector in X–Z plane (degrees)."""
    x, z = v[0], v[2]
    return float(np.degrees(np.arctan2(z, x)))


def run(ctx: Context) -> None:
    """
    Back Foot Contact (BFC) landing angle.
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
        ANKLE, HEEL, TOE = 28, 30, 32
    else:
        ANKLE, HEEL, TOE = 27, 29, 31

    ankle_y = []
    for i in range(ffc_idx):
        lm = pose_frames[i].landmarks
        ankle_y.append(float(lm[ANKLE]["y"]) if lm else 1.0)

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
    if pf.landmarks is None:
        return

    lm = pf.landmarks
    heel = np.array([lm[HEEL]["x"], lm[HEEL]["y"], lm[HEEL]["z"]])
    toe = np.array([lm[TOE]["x"], lm[TOE]["y"], lm[TOE]["z"]])

    angle = abs(_angle_2d(toe - heel))

    if angle < 25:
        zone = "CLOSED"
    elif angle < 45:
        zone = "NEUTRAL"
    elif angle < 65:
        zone = "OPEN"
    else:
        zone = "VERY_OPEN"

    # Correct reverse-anchored event write
    ctx.events.bfc = EventFrame(
        frame=bfc_frame,
        conf=float(ctx.events.ffc.conf),
    )

    ctx.biomech.backfoot = {
        "landing_angle_deg": round(angle, 2),
        "zone": zone,
        "confidence": round(float(ctx.events.ffc.conf), 2),
        "frame": bfc_frame,
        "type": "touch",
    }

    log(f"[DEBUG] BFC @frame={bfc_frame}, angle={angle:.2f}, zone={zone}")

