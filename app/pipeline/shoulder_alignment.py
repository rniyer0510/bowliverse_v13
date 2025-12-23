import numpy as np
from math import atan2, degrees
from app.utils.landmarks import LandmarkMapper

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

def run(ctx, window=7, vis_min=0.5):
    evt = ctx.events.uah
    if not evt:
        return

    mapper = LandmarkMapper(ctx.input.hand)
    frames = ctx.pose.frames

    angles, weights = [], []

    for i in range(max(0, evt.frame - window), min(len(frames), evt.frame + window + 1)):
        pf = frames[i]
        if pf.landmarks is None:
            continue

        try:
            LS, RS = mapper.shoulders_pair(pf.landmarks)
            lv = pf.landmarks[mapper.left["shoulder"]]["vis"]
            rv = pf.landmarks[mapper.right["shoulder"]]["vis"]
            vis = min(lv, rv)
        except Exception:
            continue

        if vis < vis_min:
            continue

        ang = degrees(atan2(RS[1] - LS[1], RS[0] - LS[0]))
        angles.append(ang)
        weights.append(vis)

    mean, std = weighted_stats(angles, weights)
    if mean is None:
        return

    ctx.biomech.shoulder = {
        "angle_deg": mean,
        "uncertainty_deg": std,
        "support_frames": len(angles),
        "frame": evt.frame,
    }
