# app/utils/angles.py

import numpy as np


# -----------------------------------------------------------
# GENERIC ANGLE (BACKWARD COMPATIBILITY)
# -----------------------------------------------------------

def angle(a, b, c):
    """
    Generic angle ABC using vectors BA and BC.
    """
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-9
    val = float(np.dot(ab, cb) / denom)
    val = max(-1.0, min(1.0, val))
    return float(np.degrees(np.arccos(val)))


# -----------------------------------------------------------
# BASEBALL-STYLE ANATOMICAL ELBOW FLEXION
# -----------------------------------------------------------

def elbow_flexion(shoulder, elbow, wrist):
    """
    Returns anatomical elbow flexion in degrees.
    0°   = straight arm
    160° = fully bent arm
    """
    humerus = shoulder - elbow
    forearm = wrist - elbow

    denom = (np.linalg.norm(humerus) * np.linalg.norm(forearm)) + 1e-9
    cosang = np.dot(humerus, forearm) / denom
    cosang = max(-1.0, min(1.0, cosang))

    external = float(np.degrees(np.arccos(cosang)))
    flex = 180.0 - external

    # Safe biomechanical clamp
    if flex < 0:
        flex = 0.0
    elif flex > 165:
        flex = 165.0

    return float(flex)


# -----------------------------------------------------------
# GAUSSIAN SMOOTHING
# -----------------------------------------------------------

def gaussian_smooth(values, sigma=1.0):
    """
    Smooth a sequence using a Gaussian kernel.
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
        for offset, w in zip(xs, kernel):
            j = i + offset
            if 0 <= j < N:
                acc += values[j] * w
                wsum += w
        smoothed[i] = acc / (wsum + 1e-9)

    return smoothed.tolist()


# -----------------------------------------------------------
# NEW: SEMANTIC PAIR ANGLE SERIES (LONG-TERM FIX)
# -----------------------------------------------------------

def compute_pair_angle_series(
    pose_frames,
    mapper,
    start_f,
    end_f,
    pair_fn,
):
    """
    Compute planar angle series for a bilateral joint pair
    (e.g. hips or shoulders) using LandmarkMapper semantics.

    pair_fn must be:
      - mapper.hips_pair
      - mapper.shoulders_pair

    This eliminates all 'left_*' / 'right_*' string misuse.
    """

    angles = []

    for f in range(start_f, end_f + 1):
        pf = pose_frames[f]

        if pf.landmarks is None:
            angles.append(angles[-1] if angles else 0.0)
            continue

        try:
            A, B = pair_fn(pf.landmarks)
        except Exception:
            angles.append(angles[-1] if angles else 0.0)
            continue

        dx = B[0] - A[0]
        dy = B[1] - A[1]

        angles.append(float(np.degrees(np.arctan2(dy, dx))))

    return np.array(angles, dtype=np.float32)

