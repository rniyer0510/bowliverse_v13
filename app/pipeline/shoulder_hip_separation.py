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
    evt = ctx.events.ffc
    if not evt:
        return

    mapper = LandmarkMapper(ctx.input.hand)
    frames = ctx.pose.frames

    seps, weights = [], []

    for i in range(max(0, evt.frame - window), min(len(frames), evt.frame + window + 1)):
        pf = frames[i]
        if pf.landmarks is None:
            continue

        try:
            LS, RS = mapper.shoulders_pair(pf.landmarks)
            LH, RH = mapper.hips_pair(pf.landmarks)
            sv = min(
                pf.landmarks[mapper.left["shoulder"]]["vis"],
                pf.landmarks[mapper.right["shoulder"]]["vis"],
            )
            hv = min(
                pf.landmarks[mapper.left["hip"]]["vis"],
                pf.landmarks[mapper.right["hip"]]["vis"],
            )
            vis = min(sv, hv)
        except Exception:
            continue

        if vis < vis_min:
            continue

        sh = degrees(atan2(RS[1] - LS[1], RS[0] - LS[0]))
        hp = degrees(atan2(RH[1] - LH[1], RH[0] - LH[0]))

        seps.append(sh - hp)
        weights.append(vis)

    mean, std = weighted_stats(seps, weights)
    if mean is None:
        return

    ctx.biomech.shoulder_hip = {
        "separation_deg": mean,
        "uncertainty_deg": std,
        "support_frames": len(seps),
        "frame": evt.frame,
    }
