# app/pipeline/backfoot_landing_angle.py
"""
ActionLab / Bowliverse v13.9 — BACK-FOOT LANDING ANGLE (FFC-ANCHORED, REVERSE SEARCH)

What this computes:
- Back-foot landing ANGLE only (no BFC event)
- Search direction: backward from FFC
- Detects first transition from AIR → CONTACT
- Computes foot orientation at that landing frame
- Camera-robust (ground-plane projection)

Design rules (LOCKED):
- FFC is the anchor
- Search window is strictly BEFORE FFC
- No BFC detection after FFC
- No dependency on UAH
- Graceful degradation if detection fails
"""

import numpy as np
from app.models.context import Context


# ---------------------------------------------------------
# Utility: angle in degrees between 2D vectors
# ---------------------------------------------------------
def _angle_2d(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(ctx: Context) -> None:
    """
    Stores result in ctx.biomech.backfoot

    Output structure:
    {
        "angle_deg": float,
        "zone": "CLOSED" | "NEUTRAL" | "OPEN",
        "confidence": float,
        "frame": int
    }
    """

    # -------------------------------------------------
    # Preconditions
    # -------------------------------------------------
    if not ctx.events or not ctx.events.ffc:
        return

    ffc_frame = ctx.events.ffc.frame
    ffc_conf = ctx.events.ffc.conf

    pose_frames = ctx.pose.frames
    if ffc_frame <= 2 or ffc_frame >= len(pose_frames):
        return

    # MediaPipe indices
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_TOE = 31
    RIGHT_TOE = 32

    # Decide back foot based on bowling hand
    # Right-arm bowler → right foot is back foot
    if ctx.input.hand == "R":
        heel_idx = RIGHT_HEEL
        toe_idx = RIGHT_TOE
    else:
        heel_idx = LEFT_HEEL
        toe_idx = LEFT_TOE

    # -------------------------------------------------
    # Step 1: find AIR → CONTACT transition (reverse)
    # -------------------------------------------------
    contact_frame = None
    air_threshold = 0.02      # relative vertical lift
    min_air_frames = 2        # must be airborne before landing

    was_air_count = 0

    for i in range(ffc_frame - 1, 0, -1):
        pf = pose_frames[i]
        if pf.landmarks is None:
            was_air_count = 0
            continue

        try:
            heel_y = pf.landmarks[heel_idx]["y"]
        except Exception:
            was_air_count = 0
            continue

        # Heuristic: higher Y → foot lifted (MediaPipe y increases downward)
        if heel_y > pose_frames[ffc_frame].landmarks[heel_idx]["y"] + air_threshold:
            was_air_count += 1
        else:
            if was_air_count >= min_air_frames:
                contact_frame = i
                break
            was_air_count = 0

    if contact_frame is None:
        return

    # -------------------------------------------------
    # Step 2: compute foot orientation at landing
    # -------------------------------------------------
    pf = pose_frames[contact_frame]
    if pf.landmarks is None:
        return

    try:
        heel = pf.landmarks[heel_idx]
        toe = pf.landmarks[toe_idx]

        heel_pt = np.array([heel["x"], heel["y"], heel["z"]], float)
        toe_pt = np.array([toe["x"], toe["y"], toe["z"]], float)
    except Exception:
        return

    foot_vec = toe_pt - heel_pt
    foot_xz = np.array([foot_vec[0], foot_vec[2]], float)

    ref_axis = np.array([1.0, 0.0], float)
    ang = _angle_2d(foot_xz, ref_axis)
    if ang is None:
        return

    # Handedness normalization
    if ctx.input.hand == "L":
        ang = 180.0 - ang

    # Normalize to [0, 90]
    ang = abs(ang)
    if ang > 90.0:
        ang = 180.0 - ang

    # -------------------------------------------------
    # Zone classification
    # -------------------------------------------------
    if ang < 20.0:
        zone = "CLOSED"
    elif ang < 45.0:
        zone = "NEUTRAL"
    else:
        zone = "OPEN"

    # -------------------------------------------------
    # Store result
    # -------------------------------------------------
    ctx.biomech.backfoot = {
        "angle_deg": round(float(ang), 2),
        "zone": zone,
        "confidence": round(float(ffc_conf), 2),
        "frame": int(contact_frame),
    }

