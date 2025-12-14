# app/pipeline/backfoot_contact.py

import numpy as np
from app.models.context import Context
from app.utils.angles import angle


def run(ctx: Context) -> None:
    """
    Back-foot orientation at Back Foot Contact (BFC).
    Produces an intent zone (SIDE_ON / TRANSITIONAL / FRONT_ON).

    NOTE:
    - Uses existing angles.angle(a,b,c) to stay consistent with the codebase.
    - Uses pose_frames as a LIST (ctx.pose.frames[index]) — aligned with current pipeline.
    - Pitch reference here is +Y as a placeholder; we will calibrate axis later if needed.
    """

    if not ctx.events or not ctx.events.bfc:
        return

    frame = ctx.events.bfc.frame
    conf = ctx.events.bfc.conf

    pose_frames = ctx.pose.frames
    if frame < 0 or frame >= len(pose_frames):
        return

    pf = pose_frames[frame]
    if pf.landmarks is None:
        return

    lm = pf.landmarks

    # MediaPipe indices
    LEFT_HEEL, LEFT_TOE = 29, 31
    RIGHT_HEEL, RIGHT_TOE = 30, 32

    is_right_hander = ctx.input.hand.upper() == "R"
    heel = lm[RIGHT_HEEL] if is_right_hander else lm[LEFT_HEEL]
    toe  = lm[RIGHT_TOE]  if is_right_hander else lm[LEFT_TOE]

    heel_pt = np.array([heel["x"], heel["y"], heel["z"]], float)
    toe_pt  = np.array([toe["x"],  toe["y"],  toe["z"]],  float)

    pitch_pt = heel_pt + np.array([0.0, 1.0, 0.0])  # forward reference

    ang = float(angle(toe_pt, heel_pt, pitch_pt))

    # Practical zones (can be tuned later)
    if ang <= 25:
        zone = "SIDE_ON"
    elif ang <= 45:
        zone = "TRANSITIONAL"
    else:
        zone = "FRONT_ON"

    ctx.biomech.backfoot = {
        "angle_deg": round(ang, 2),
        "zone": zone,
        "confidence": round(float(conf), 2),
        "frame": int(frame),
    }
