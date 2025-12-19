# app/pipeline/ffc_reverse.py

import numpy as np
from app.models.events_model import EventFrame
from app.utils.logger import log

# MediaPipe indices
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30


def _is_grounded(lm, ankle_idx, heel_idx, eps=0.015):
    """
    Foot considered grounded if:
    - ankle and heel exist
    - vertical separation is small (flat + stable)
    """
    try:
        ay = lm[ankle_idx]["y"]
        hy = lm[heel_idx]["y"]
        return abs(ay - hy) < eps
    except Exception:
        return False


def detect_ffc_reverse(pose_frames, anchor_idx, hand, min_stable=3):
    """
    Reverse FFC detection (STABILITY-BASED).

    Logic:
    - Start from anchor frame (UAH)
    - Go backwards
    - Identify a *stable grounded window*
    - Return the FIRST frame of that stable window
    """

    if anchor_idx <= min_stable + 2:
        return None

    # Front foot = opposite of bowling arm
    if hand.upper() == "R":
        ankle = LEFT_ANKLE
        heel = LEFT_HEEL
    else:
        ankle = RIGHT_ANKLE
        heel = RIGHT_HEEL

    stable_frames = []

    for f in range(anchor_idx, 1, -1):
        pf = pose_frames[f]
        if pf.landmarks is None:
            stable_frames.clear()
            continue

        grounded = _is_grounded(pf.landmarks, ankle, heel)

        if grounded:
            stable_frames.append(f)

            # Once we have enough stable frames,
            # keep going back to find where stability STARTED
            if len(stable_frames) >= min_stable:
                continue

        else:
            # We just exited a stable region → FFC detected
            if len(stable_frames) >= min_stable:
                ffc_frame = max(stable_frames)
                conf = float(pose_frames[ffc_frame].confidence or 0.8) * 100.0

                log(f"[INFO] Reverse FFC (stable) detected @ frame={ffc_frame}")
                return EventFrame(frame=ffc_frame, conf=conf)

            stable_frames.clear()

    log("[WARN] Reverse FFC not reliably detected")
    return None

