# app/pipeline/backfoot_landing_angle.py

import numpy as np
from math import atan2, degrees
from app.utils.landmarks import LandmarkMapper
from app.utils.logger import log


def weighted_stats(values, weights):
    if not values:
        return None, None
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    s = np.sum(w)
    if s <= 0:
        return None, None
    mean = np.sum(w * v) / s
    var = np.sum(w * (v - mean) ** 2) / s
    return float(mean), float(np.sqrt(var))


def run(ctx, window=5, vis_min=0.5):
    """
    Back-foot landing angle at BFC.

    Notes:
    - Pure measurement stage
    - NO zoning or semantic interpretation here
    - Toe → heel vector
    - Interpretation happens downstream
    """

    evt = ctx.events.bfc
    if not evt:
        return

    mapper = LandmarkMapper(ctx.input.hand)
    frames = ctx.pose.frames

    angles = []
    weights = []

    for i in range(
        max(0, evt.frame - window),
        min(len(frames), evt.frame + window + 1),
    ):
        pf = frames[i]
        if pf.landmarks is None:
            continue

        try:
            toe = mapper.vec(pf.landmarks, "toe")
            heel = mapper.vec(pf.landmarks, "heel")
            vis = min(
                pf.landmarks[mapper.primary["toe"]]["vis"],
                pf.landmarks[mapper.primary["heel"]]["vis"],
            )
        except Exception:
            continue

        if vis < vis_min:
            continue

        ang = degrees(atan2(toe[1] - heel[1], toe[0] - heel[0]))
        angles.append(ang)
        weights.append(vis)

    mean, std = weighted_stats(angles, weights)
    if mean is None:
        log("[WARN] Backfoot landing angle skipped: insufficient data")
        return

    ctx.biomech.backfoot = {
        "angle_deg": round(mean, 2),
        "uncertainty_deg": round(std or 0.0, 2),
        "support_frames": len(angles),
        "frame": evt.frame,
        "confidence": evt.conf,
    }

