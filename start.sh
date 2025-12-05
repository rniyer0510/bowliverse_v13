#!/bin/bash
set -e

echo "📌 Applying Bowliverse v13.7 Baseball Fix…"

# -------------------------
# Update app/utils/angles.py
# -------------------------
cat > app/utils/angles.py << 'EOF'
import numpy as np
import math


def angle(a, b, c):
    """
    Generic angle ABC in degrees using vectors BA and BC.
    Retained for backward compatibility.
    """
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-9
    val = float(np.dot(ab, cb) / denom)
    val = max(-1.0, min(1.0, val))
    return float(np.degrees(np.arccos(val)))


# -------------------------------------------------------------------
# BASEBALL-STYLE PROJECTION-BASED ELBOW FLEXION
# -------------------------------------------------------------------

def elbow_flexion_projected(shoulder, elbow, wrist, plane_normal):
    """
    Anatomical elbow flexion using projection into the arm plane.

    0°   = full extension
    160° = fully bent

    Steps:
        1) Compute humerus & forearm vectors
        2) Project both vectors into detected arm plane
        3) Compute external angle
        4) flexion = 180 - external_angle
    """
    humerus = shoulder - elbow
    forearm = wrist - elbow

    # Project into the arm plane
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)

    hum_proj = humerus - np.dot(humerus, n) * n
    wr_proj  = forearm - np.dot(forearm, n) * n

    # Angle between projected segments
    denom = (np.linalg.norm(hum_proj) * np.linalg.norm(wr_proj)) + 1e-9
    cosang = float(np.dot(hum_proj, wr_proj) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    external = float(np.degrees(np.arccos(cosang)))

    flex = 180.0 - external
    flex = max(0.0, min(flex, 165.0))
    return flex


# Gaussian smoothing
def gaussian_smooth(values, sigma=1.0):
    """
    Smooth noisy flexion curves.
    """
    if len(values) <= 2:
        return values[:]

    N = len(values)
    smoothed = np.zeros(N, float)

    radius = int(3 * sigma)
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    for i in range(N):
        acc = 0.0
        wsum = 0.0
        for k, w in zip(xs, kernel):
            j = i + k
            if 0 <= j < N:
                acc += values[j] * w
                wsum += w
        smoothed[i] = acc / (wsum + 1e-9)

    return smoothed.tolist()
EOF


# -------------------------------
# Update app/utils/landmarks.py
# -------------------------------
cat > app/utils/landmarks.py << 'EOF'
import numpy as np


class LandmarkMapper:
    """
    Bowliverse v13.7 — Landmark Mapper + Baseball Plane Detector
    """

    def __init__(self, hand: str, vis_threshold: float = 0.20):
        self.hand = hand.upper()
        self.vis_threshold = vis_threshold

        self.left = {"shoulder": 11, "elbow": 13, "wrist": 15,
                     "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31}

        self.right = {"shoulder": 12, "elbow": 14, "wrist": 16,
                      "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "toe": 32}

        self.primary = self.right if self.hand == "R" else self.left

    def _safe_vec(self, lm, idx, fallback=None):
        p = lm[idx]
        vis = float(p.get("vis", 0.0))
        if vis < self.vis_threshold and fallback is not None:
            return fallback
        return np.array([p["x"], p["y"], p["z"]], float)

    # ---------- Dominant arm triplet ----------
    def arm_triplet(self, lm, prev=None):
        prev_w = prev["w"] if prev else None
        prev_e = prev["e"] if prev else None
        prev_s = prev["s"] if prev else None
        W = self._safe_vec(lm, self.primary["wrist"], fallback=prev_w)
        E = self._safe_vec(lm, self.primary["elbow"], fallback=prev_e)
        S = self._safe_vec(lm, self.primary["shoulder"], fallback=prev_s)
        return W, E, S

    # ---------- Hips + Shoulders ----------
    def hips_pair(self, lm):
        return (
            self._safe_vec(lm, self.left["hip"]),
            self._safe_vec(lm, self.right["hip"])
        )

    def shoulders_pair(self, lm):
        return (
            self._safe_vec(lm, self.left["shoulder"]),
            self._safe_vec(lm, self.right["shoulder"])
        )

    # ---------- Arm-plane normal (Baseball method) ----------
    def arm_plane_normal(self, lm):
        LS, RS = self.shoulders_pair(lm)
        LH, RH = self.hips_pair(lm)

        shoulder_axis = RS - LS
        hip_axis = RH - LH
        trunk_axis = shoulder_axis + hip_axis
        trunk_axis = trunk_axis / (np.linalg.norm(trunk_axis) + 1e-9)

        global_up = np.array([0.0, 1.0, 0.0], float)

        # normal to the arm motion plane
        n = np.cross(trunk_axis, global_up)
        return n / (np.linalg.norm(n) + 1e-9)

    def vec(self, lm, key: str):
        idx = self.primary.get(key)
        return self._safe_vec(lm, idx)
EOF


# ---------------------------------------
# Update app/pipeline/biomech_stage.py
# ---------------------------------------
cat > app/pipeline/biomech_stage.py << 'EOF'
import numpy as np

from app.models.context import Context
from app.models.biomech_model import BiomechElbowModel, BiomechReleaseHeightModel
from app.utils.landmarks import LandmarkMapper
from app.utils.angles import elbow_flexion_projected, gaussian_smooth


def _visibility(mapper, lm):
    idxs = [mapper.primary["shoulder"], mapper.primary["elbow"], mapper.primary["wrist"]]
    vals = [float(lm[j].get("vis", 0.0)) for j in idxs]
    return float(sum(vals) / len(vals))


def run(ctx: Context) -> Context:
    events = ctx.events
    frames = ctx.pose.frames

    if events.release is None or events.uah is None:
        ctx.biomech.error = "Missing UAH or Release frame."
        return ctx

    rel_idx = events.release.frame
    uah_idx = events.uah.frame
    if rel_idx >= len(frames) or uah_idx >= len(frames):
        ctx.biomech.error = "Event frame index out of bounds."
        return ctx

    if uah_idx > rel_idx:
        uah_idx, rel_idx = rel_idx, uah_idx

    mapper = LandmarkMapper(ctx.input.hand)

    flexions = []
    for i in range(uah_idx, rel_idx + 1):
        lm = frames[i].landmarks
        W, E, S = mapper.arm_triplet(lm)
        n = mapper.arm_plane_normal(lm)
        flex = elbow_flexion_projected(S, E, W, n)
        flexions.append(flex)

    flex_smooth = gaussian_smooth(flexions, sigma=1.0)

    flex_min = float(min(flex_smooth))
    flex_max = float(max(flex_smooth))
    excursion = max(0.0, flex_max - flex_min)

    uah_flex = float(flex_smooth[0])
    rel_flex = float(flex_smooth[-1])

    # Release height
    rel_frame = frames[rel_idx]
    LS, RS = mapper.shoulders_pair(rel_frame.landmarks)
    LH, RH = mapper.hips_pair(rel_frame.landmarks)

    shoulder_mid = (LS + RS) / 2.0
    hip_mid = (LH + RH) / 2.0
    torso_len = abs(shoulder_mid[1] - hip_mid[1]) + 1e-6

    W_rel, _, _ = mapper.arm_triplet(rel_frame.landmarks)
    wrist_y = float(W_rel[1])
    norm_height = float((shoulder_mid[1] - wrist_y) / torso_len)

    vis_rel = _visibility(mapper, rel_frame.landmarks)
    vis_uah = _visibility(mapper, frames[uah_idx].landmarks)

    elbow_conf = int(
        50 * ((rel_frame.confidence + frames[uah_idx].confidence) / 2.0)
        + 50 * ((vis_rel + vis_uah) / 2.0)
    )
    elbow_conf = max(0, min(elbow_conf, 100))

    release_height_conf = int(100 * rel_frame.confidence)
    release_height_conf = max(0, min(release_height_conf, 100))

    if elbow_conf > 85:
        note = "High confidence estimate"
    elif elbow_conf > 60:
        note = "Moderate confidence; minor occlusion"
    else:
        note = "Low confidence; visibility issues"

    ctx.biomech.elbow = BiomechElbowModel(
        uah_angle=uah_flex,
        release_angle=rel_flex,
        extension_deg=excursion,
        extension_raw_deg=excursion,
        extension_error_margin_deg=6.0,
        extension_note=note,
    )

    ctx.biomech.release_height = BiomechReleaseHeightModel(
        norm_height=norm_height,
        wrist_y=wrist_y,
    )

    ctx.biomech.elbow_conf = elbow_conf
    ctx.biomech.release_height_conf = release_height_conf

    ctx.biomech.angle_plane = {
        "uah_plane": "projected",
        "release_plane": "projected",
    }

    ctx.biomech.error = None
    return ctx
EOF

echo "✅ Baseball Fix applied successfully!"

