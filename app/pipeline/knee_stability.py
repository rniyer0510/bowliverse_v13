import numpy as np


def _angle(a, b, c):
    ab = a - b
    cb = c - b
    dot = np.dot(ab, cb)
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-6
    return np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))


def classify_knee_stability(frames, hip_idx, knee_idx, ankle_idx):
    """
    Front-knee stability classifier (Phase-1 conservative).

    Strategy:
    - If signal confidence is low → return Inconclusive
    - Avoid false positives from monocular noise
    """

    # -----------------------------
    # Confidence gating
    # -----------------------------
    if frames is None or len(frames) < 4:
        return "Inconclusive", "Low"

    angles = []

    for lm in frames:
        try:
            hip = np.array([lm[hip_idx]["x"], lm[hip_idx]["y"], lm[hip_idx]["z"]])
            knee = np.array([lm[knee_idx]["x"], lm[knee_idx]["y"], lm[knee_idx]["z"]])
            ankle = np.array([lm[ankle_idx]["x"], lm[ankle_idx]["y"], lm[ankle_idx]["z"]])
            angles.append(_angle(hip, knee, ankle))
        except Exception:
            continue

    # If too few reliable measurements → inconclusive
    if len(angles) < 4:
        return "Inconclusive", "Low"

    delta = max(angles) - min(angles)

    # Large delta in monocular setup is unreliable → inconclusive
    if delta > 20:
        return "Inconclusive", "Low"

    # Conservative classification only if signal is stable
    if delta <= 12:
        return "Stable", "Low"
    elif delta <= 18:
        return "Soft brace", "Low"
    else:
        return "Inconclusive", "Low"
