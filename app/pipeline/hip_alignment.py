# app/pipeline/hip_alignment.py

import numpy as np
from app.models.context import Context
from app.utils.angles import angle


WINDOW = 2  # frames around UAH


def run(ctx: Context) -> None:
    """
    Hip orientation near UAH (action classification phase).

    Camera-agnostic:
    - Hip line: LEFT_HIP → RIGHT_HIP
    - Forward reference: hip-midpoint → shoulder-midpoint

    Output: ctx.biomech.hip = {angle_deg, zone, confidence, frame}
    """

    events = ctx.events
    if not events or not events.uah:
        return

    center = events.uah.frame
    conf = events.uah.conf

    pose_frames = ctx.pose.frames
    if center < 0 or center >= len(pose_frames):
        return

    # Use small window around UAH for stability
    frames = [
        f for f in range(center - WINDOW, center + WINDOW + 1)
        if 0 <= f < len(pose_frames)
    ]

    angles = []

    for f in frames:
        pf = pose_frames[f]
        if pf.landmarks is None:
            continue

        lm = pf.landmarks

        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        try:
            Lh = np.array([lm[LEFT_HIP]["x"], lm[LEFT_HIP]["y"], lm[LEFT_HIP]["z"]])
            Rh = np.array([lm[RIGHT_HIP]["x"], lm[RIGHT_HIP]["y"], lm[RIGHT_HIP]["z"]])
            Ls = np.array([lm[LEFT_SHOULDER]["x"], lm[LEFT_SHOULDER]["y"], lm[LEFT_SHOULDER]["z"]])
            Rs = np.array([lm[RIGHT_SHOULDER]["x"], lm[RIGHT_SHOULDER]["y"], lm[RIGHT_SHOULDER]["z"]])
        except Exception:
            continue

        hip_mid = (Lh + Rh) / 2.0
        shoulder_mid = (Ls + Rs) / 2.0

        hip_vec = Rh - Lh
        body_vec = shoulder_mid - hip_mid

        if np.linalg.norm(hip_vec) < 1e-6 or np.linalg.norm(body_vec) < 1e-6:
            continue

        ang = float(angle(Rh, hip_mid, hip_mid + body_vec))
        angles.append(ang)

    if not angles:
        return

    ang = float(np.median(angles))

    # Broad zones (intentionally overlapping)
    if ang <= 35:
        zone = "SIDE_ON"
    elif ang <= 60:
        zone = "MIXED"
    else:
        zone = "FRONT_ON"

    ctx.biomech.hip = {
        "angle_deg": round(ang, 2),
        "zone": zone,
        "confidence": round(float(conf), 2),
        "frame": int(center),
    }

