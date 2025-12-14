# app/pipeline/shoulder_hip_separation.py

import numpy as np
from app.models.context import Context
from app.utils.logger import log


# MediaPipe landmark indices (image-space)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def _line_angle(p1, p2):
    """
    Return orientation angle of a line (p1 -> p2) in degrees.
    Normalized to [0, 180).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    ang = abs(np.degrees(np.arctan2(dy, dx)))
    if ang >= 180:
        ang -= 180
    return ang


def run(ctx: Context) -> Context:
    """
    Shoulder–Hip separation (torsion) at FFC.
    Camera-relative but largely yaw-invariant.

    Output:
        ctx.biomech.shoulder_hip = {
            "angle_deg": float,
            "zone": "STACKED" | "MILD" | "MODERATE" | "HIGH",
            "confidence": float,
            "frame": int
        }
    """

    events = ctx.events
    pose_frames = ctx.pose.frames

    ctx.biomech.shoulder_hip = None

    if not events.ffc:
        log("[DEBUG] ShoulderHipSep: FFC not available, skipping")
        return ctx

    f_ffc = events.ffc.frame
    if not (0 <= f_ffc < len(pose_frames)):
        log("[DEBUG] ShoulderHipSep: Invalid FFC frame index")
        return ctx

    # Small temporal window for stability
    window = range(max(0, f_ffc - 2), min(len(pose_frames), f_ffc + 3))

    sep_angles = []
    confs = []

    for i in window:
        pf = pose_frames[i]
        if pf.landmarks is None:
            continue

        lm = pf.landmarks
        try:
            LS = np.array([lm[LEFT_SHOULDER]["x"], lm[LEFT_SHOULDER]["y"]])
            RS = np.array([lm[RIGHT_SHOULDER]["x"], lm[RIGHT_SHOULDER]["y"]])
            LH = np.array([lm[LEFT_HIP]["x"], lm[LEFT_HIP]["y"]])
            RH = np.array([lm[RIGHT_HIP]["x"], lm[RIGHT_HIP]["y"]])

            ang_sh = _line_angle(LS, RS)
            ang_hip = _line_angle(LH, RH)

            d = abs(ang_sh - ang_hip)
            sep = min(d, 180 - d)  # [0..90]

            sep_angles.append(sep)

            # confidence from visibility (if present)
            vis = [
                lm[LEFT_SHOULDER].get("vis", 1.0),
                lm[RIGHT_SHOULDER].get("vis", 1.0),
                lm[LEFT_HIP].get("vis", 1.0),
                lm[RIGHT_HIP].get("vis", 1.0),
            ]
            confs.append(min(vis))

        except Exception:
            continue

    if not sep_angles:
        log("[DEBUG] ShoulderHipSep: No valid frames in window")
        return ctx

    sep_deg = float(np.median(sep_angles))
    conf = float(np.median(confs)) if confs else 1.0

    if sep_deg < 15:
        zone = "STACKED"
    elif sep_deg <= 30:
        zone = "MILD"
    elif sep_deg <= 50:
        zone = "MODERATE"
    else:
        zone = "HIGH"

    ctx.biomech.shoulder_hip = {
        "angle_deg": round(sep_deg, 2),
        "zone": zone,
        "confidence": round(conf, 2),
        "frame": f_ffc,
    }

    log(
        f"[DEBUG] ShoulderHipSep @FFC={f_ffc}: "
        f"sep={sep_deg:.2f} deg, zone={zone}, conf={conf:.2f}"
    )

    return ctx

