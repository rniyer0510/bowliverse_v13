# app/pipeline/events_stage.py
"""
Bowliverse v13.5 — Event Detection Engine
-----------------------------------------

Event order:
    BFC < FFC < UAH < RELEASE

Stages:
    1) Pre-smoothed frames (from occlusion.smooth)
    2) Coarse Release (legacy wrist-Y minima)
    3) UAH (elbow-angle maximum, backward search, dynamic window)
    4) FFC (ankle-Y minima, backward search from UAH)
    5) BFC (COM-Y maxima, backward search from FFC)
    6) Release refinement (forward elbow-angle local minima from UAH)
    7) Temporal enforcement

All detectors use:
    - dominant-arm mapping (mapper.primary)
    - joint visibility for confidence
    - error-tolerant fallbacks

This engine is stable for occlusions, fast arm motion, slight camera tilt,
and v13 biomech requirements.
"""

import numpy as np
from app.models.context import Context
from app.models.events_model import EventFrame
from app.utils.landmarks import LandmarkMapper
from app.utils.angles import angle as angle_abc


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

def _percent(v: float) -> int:
    """Convert visibility-like 0–1 float to 0–100 int."""
    return int(max(0.0, min(v * 100.0, 100.0)))


def _clamp(i, n):
    return max(0, min(i, n - 1))


def _elbow_angle(mapper, lm):
    W, E, S = mapper.arm_triplet(lm)
    return float(angle_abc(W, E, S))


def _arm_vis(mapper, lm):
    idxs = [mapper.primary["shoulder"],
            mapper.primary["elbow"],
            mapper.primary["wrist"]]
    vals = []
    for j in idxs:
        if 0 <= j < len(lm):
            vals.append(lm[j].get("vis", 0.0))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


# -----------------------------------------------------
# 1. COARSE RELEASE (legacy wrist-Y minima)
# -----------------------------------------------------

def detect_coarse_release(frames, mapper):
    wrist_y = []
    for f in frames:
        W, _, _ = mapper.arm_triplet(f.landmarks)
        wrist_y.append(W[1])

    idx = int(np.argmin(wrist_y))
    conf = _percent(frames[idx].confidence)
    return idx, conf, False


# -----------------------------------------------------
# 2. UAH — backward elbow-angle maxima
# -----------------------------------------------------

def detect_uah(frames, release_idx, mapper):
    n = len(frames)

    # Dynamic window = 20% clip, clamped to 10–35
    span = max(10, int(0.20 * n))
    span = min(span, 35)

    start = _clamp(release_idx - span, n)
    end = _clamp(release_idx - 3, n)

    best_idx = None
    best_angle = -1e9
    best_vis = 0.0

    for i in range(start, end + 1):
        lm = frames[i].landmarks

        vis = _arm_vis(mapper, lm)
        if vis < 0.30:  # low visibility → discard
            continue

        ang = _elbow_angle(mapper, lm)

        if ang > best_angle or (abs(ang - best_angle) < 1.0 and vis > best_vis):
            best_angle = ang
            best_vis = vis
            best_idx = i

    fallback = False
    if best_idx is None:
        fallback = True
        best_idx = _clamp(release_idx - 6, n)
        best_vis = _arm_vis(mapper, frames[best_idx].landmarks)

    conf = _percent(best_vis)
    if fallback:
        conf = max(conf - 5, 0)

    return best_idx, conf, fallback


# -----------------------------------------------------
# 3. FFC — backward ankle-Y minima
# -----------------------------------------------------

def detect_ffc(frames, uah_idx, mapper):
    n = len(frames)
    span = max(8, int(0.15 * n))
    span = min(span, 30)

    start = _clamp(uah_idx - span, n)
    end = _clamp(uah_idx - 1, n)

    candidates = []
    for i in range(start, end + 1):
        ay = mapper.ankle_y(frames[i].landmarks)
        candidates.append((i, ay))

    if not candidates:
        idx = _clamp(uah_idx - int(span / 2), n)
        conf = max(_percent(frames[idx].confidence) - 3, 0)
        return idx, conf, True

    idx, _ = min(candidates, key=lambda x: x[1])
    conf = _percent(frames[idx].confidence)
    return idx, conf, False


# -----------------------------------------------------
# 4. BFC — backward COM-Y maxima
# -----------------------------------------------------

def detect_bfc(frames, ffc_idx, mapper):
    n = len(frames)
    span = max(10, int(0.18 * n))
    span = min(span, 35)

    start = _clamp(ffc_idx - span, n)
    end = _clamp(ffc_idx - 1, n)

    candidates = []
    for i in range(start, end + 1):
        lm = frames[i].landmarks
        lh, rh = mapper.hips_pair(lm)
        ls, rs = mapper.shoulders_pair(lm)
        com_y = ((lh + rh + ls + rs) / 4.0)[1]
        candidates.append((i, com_y))

    if not candidates:
        idx = _clamp(ffc_idx - int(span / 2), n)
        conf = max(_percent(frames[idx].confidence) - 3, 0)
        return idx, conf, True

    idx, _ = max(candidates, key=lambda x: x[1])
    conf = _percent(frames[idx].confidence)
    return idx, conf, False


# -----------------------------------------------------
# 5. FINAL RELEASE — forward elbow-angle refinement from UAH
# -----------------------------------------------------

def refine_release(frames, mapper, uah_idx, coarse_rel_idx):
    n = len(frames)

    span = max(6, int(0.15 * n))
    span = min(span, 30)

    start = _clamp(uah_idx + 1, n)
    end = _clamp(uah_idx + span, n)

    angle_uah = _elbow_angle(mapper, frames[uah_idx].landmarks)

    candidates = []

    prev_ang = None

    for i in range(start, end + 1):
        lm = frames[i].landmarks
        ang = _elbow_angle(mapper, lm)
        vis = _arm_vis(mapper, lm)

        if vis < 0.40:  # visibility gate
            prev_ang = ang
            continue

        if prev_ang is not None and ang > prev_ang:
            prev_ang = ang
            continue

        if ang >= angle_uah:
            prev_ang = ang
            continue

        candidates.append((i, ang, vis))
        prev_ang = ang

    if not candidates:
        conf = _percent(frames[coarse_rel_idx].confidence)
        return coarse_rel_idx, conf, True

    def key(x):
        idx, ang, vis = x
        return (ang, -vis)

    best_idx, best_ang, best_vis = min(candidates, key=key)

    angle_gap = max(0.0, angle_uah - best_ang)
    gap_norm = min(angle_gap / 20.0, 1.0)
    conf_norm = (0.5 * best_vis) + (0.5 * gap_norm)
    conf = int(round(conf_norm * 100.0))

    return best_idx, conf, False


# -----------------------------------------------------
# Temporal enforcement
# -----------------------------------------------------

def temporal_fix(bfc, ffc, uah, rel):
    if bfc >= ffc:
        bfc = max(0, ffc - 2)
    if ffc >= uah:
        ffc = max(0, uah - 2)
    if uah >= rel:
        uah = max(0, rel - 2)
    return bfc, ffc, uah, rel


# -----------------------------------------------------
# MAIN ENTRY
# -----------------------------------------------------

def run(ctx: Context) -> Context:
    frames = ctx.pose.frames
    if not frames:
        ctx.events.error = "No pose frames available"
        return ctx

    n = len(frames)
    mapper = LandmarkMapper(ctx.input.hand)

    # 1) coarse release
    coarse_rel_idx, coarse_rel_conf, _coarse_fb = detect_coarse_release(frames, mapper)

    # 2) UAH
    uah_idx, uah_conf, _uah_fb = detect_uah(frames, coarse_rel_idx, mapper)

    # 3) FFC
    ffc_idx, ffc_conf, _ffc_fb = detect_ffc(frames, uah_idx, mapper)

    # 4) BFC
    bfc_idx, bfc_conf, _bfc_fb = detect_bfc(frames, ffc_idx, mapper)

    # 5) refined release
    final_rel_idx, final_rel_conf, _rel_fb = refine_release(
        frames, mapper, uah_idx, coarse_rel_idx
    )

    # clamp safety
    bfc_idx = _clamp(bfc_idx, n)
    ffc_idx = _clamp(ffc_idx, n)
    uah_idx = _clamp(uah_idx, n)
    final_rel_idx = _clamp(final_rel_idx, n)

    # temporal correction
    bfc_idx, ffc_idx, uah_idx, final_rel_idx = temporal_fix(
        bfc_idx, ffc_idx, uah_idx, final_rel_idx
    )

    # Save Events
    ctx.events.bfc = EventFrame(frame=bfc_idx, conf=bfc_conf)
    ctx.events.ffc = EventFrame(frame=ffc_idx, conf=ffc_conf)
    ctx.events.uah = EventFrame(frame=uah_idx, conf=uah_conf)
    ctx.events.release = EventFrame(frame=final_rel_idx, conf=final_rel_conf)
    ctx.events.error = None

    return ctx

