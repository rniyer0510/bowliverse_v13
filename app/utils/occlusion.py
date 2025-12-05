# app/utils/occlusion.py
import copy
from statistics import median


def _median_visibility(lm):
    """Median visibility of all 33 landmarks."""
    return median(p.get("vis", 0.0) for p in lm)


def _repair_landmarks(prev_lm, curr_lm, vis_threshold=0.15):
    """
    Joint-level occlusion repair:
    If a landmark's visibility < threshold, copy previous frame's value.
    Does NOT overwrite well-tracked joints, preserving true motion.

    prev_lm, curr_lm are lists of 33 dicts.
    Returns repaired copy of curr_lm.
    """
    repaired = copy.deepcopy(curr_lm)

    for i in range(len(curr_lm)):
        if curr_lm[i].get("vis", 0.0) < vis_threshold:
            # fallback to previous frame
            repaired[i]["x"] = prev_lm[i]["x"]
            repaired[i]["y"] = prev_lm[i]["y"]
            repaired[i]["z"] = prev_lm[i]["z"]
            repaired[i]["vis"] = prev_lm[i].get("vis", 0.0)

    return repaired


def smooth(frames, vis_frame_threshold=0.20, vis_joint_threshold=0.15):
    """
    Hybrid Occlusion Handling (v13.5)

    1) FRAME-LEVEL CHECK:
       If median vis < vis_frame_threshold → replace frame with previous frame.
       (Useful for complete loss due to motion blur.)

    2) JOINT-LEVEL REPAIR:
       For each joint with low visibility → copy previous valid joint.
       (Preserves geometry while fixing only corrupted points.)

    Notes:
    - NEVER destroys correct frames.
    - Ensures stability for:
        • UAH elbow angle
        • Release refinement
        • FFC ankle detection
        • BFC COM detection

    frames: list of PoseFrame
    Returns: list of repaired PoseFrame
    """

    if not frames:
        return frames

    # Clone to avoid mutating original
    out = copy.deepcopy(frames)

    # -------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------
    for i in range(1, len(out)):
        prev = out[i - 1]
        curr = out[i]

        # Frame-level occlusion check
        if _median_visibility(curr.landmarks) < vis_frame_threshold:
            # Replace frame entirely
            out[i].landmarks = copy.deepcopy(prev.landmarks)
            out[i].confidence = prev.confidence
            continue

        # Joint-level repair
        repaired_landmarks = _repair_landmarks(
            prev.landmarks,
            curr.landmarks,
            vis_threshold=vis_joint_threshold,
        )
        out[i].landmarks = repaired_landmarks

    # -------------------------------------------------------
    # Backward smoothing pass
    # -------------------------------------------------------
    for i in range(len(out) - 2, -1, -1):
        curr = out[i]
        nxt = out[i + 1]

        if _median_visibility(curr.landmarks) < vis_frame_threshold:
            out[i].landmarks = copy.deepcopy(nxt.landmarks)
            out[i].confidence = nxt.confidence
            continue

        repaired_landmarks = _repair_landmarks(
            nxt.landmarks,
            curr.landmarks,
            vis_threshold=vis_joint_threshold,
        )
        out[i].landmarks = repaired_landmarks

    return out

