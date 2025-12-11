"""
Bowliverse v13.8 — EVENTS STAGE (C-2 Adaptive UAH Refinement)
--------------------------------------------------------------

Goal:
- Keep event semantics intact (Release → UAH → FFC → BFC).
- Improve UAH detection by combining:
    • flexion minima
    • humerus elevation
    • shoulder rotation velocity
- Bounded correction (±8 frames)
- Occlusion-safe with minimum visibility filter
"""

from typing import Optional
import numpy as np

from app.pipeline.context import Context
from app.utils.landmarks import LandmarkMapper
from app.models.events_model import EventFrame


# -----------------------------------------------------
# Helper: smoothing
# -----------------------------------------------------
def _smooth(vals):
    if len(vals) < 3:
        return vals[:]
    return np.convolve(vals, [0.25, 0.5, 0.25], mode="same").tolist()


# -----------------------------------------------------
# Visibility filter
# -----------------------------------------------------
def _is_valid_frame(pf, mapper, min_vis=0.15):
    lm = pf.landmarks
    if lm is None:
        return False
    try:
        sh = lm[mapper.primary["shoulder"]]["vis"]
        el = lm[mapper.primary["elbow"]]["vis"]
        wr = lm[mapper.primary["wrist"]]["vis"]
        return (sh >= min_vis and el >= min_vis and wr >= min_vis)
    except Exception:
        return False


# -----------------------------------------------------
# Compute flexion curve
# -----------------------------------------------------
def _flexion_list(pose_frames, mapper):
    flex_list = []
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            flex_list.append(None)
            continue
        try:
            S = mapper.vec(lm, "shoulder")
            E = mapper.vec(lm, "elbow")
            W = mapper.vec(lm, "wrist")
            hum = S - E
            fore = W - E
            denom = (np.linalg.norm(hum) * np.linalg.norm(fore)) + 1e-9
            cosang = np.dot(hum, fore) / denom
            cosang = max(-1.0, min(1.0, cosang))
            external = float(np.degrees(np.arccos(cosang)))
            flex = 180.0 - external
        except Exception:
            flex = None
        flex_list.append(flex)

    cleaned = [(0.0 if a is None else a) for a in flex_list]
    smoothed = _smooth(cleaned)
    return smoothed


# -----------------------------------------------------
# Compute humerus elevation (shoulder → elbow vector Y)
# -----------------------------------------------------
def _elevation_list(pose_frames, mapper):
    vals = []
    idx_sh = mapper.primary["shoulder"]
    idx_el = mapper.primary["elbow"]

    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            vals.append(None)
            continue
        try:
            sh = lm[idx_sh]["y"]
            el = lm[idx_el]["y"]
            vals.append(float(sh - el))  # positive when shoulder above elbow
        except Exception:
            vals.append(None)

    cleaned = [(0.0 if a is None else a) for a in vals]
    return _smooth(cleaned)


# -----------------------------------------------------
# Compute shoulder rotation velocity (ΔX of shoulder vs hip)
# -----------------------------------------------------
def _rotation_list(pose_frames, mapper):
    vals = []
    sh_idx = mapper.primary["shoulder"]
    hip_idx = mapper.primary["hip"]

    prev = None
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            vals.append(0.0)
            continue
        shx = lm[sh_idx]["x"]
        hix = lm[hip_idx]["x"]
        if prev is None:
            vals.append(0.0)
        else:
            vals.append(float((shx - hix) - prev))
        prev = (shx - hix)
    return _smooth(vals)


# -----------------------------------------------------
# Release detection (same as before)
# -----------------------------------------------------
def detect_release(pose_frames, mapper) -> Optional[EventFrame]:
    angles = []
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            angles.append(None)
            continue
        try:
            S = mapper.vec(lm, "shoulder")
            E = mapper.vec(lm, "elbow")
            W = mapper.vec(lm, "wrist")
            hum = S - E
            fore = W - E
            denom = (np.linalg.norm(hum) * np.linalg.norm(fore)) + 1e-9
            cosang = np.dot(hum, fore) / denom
            cosang = max(-1.0, min(1.0, cosang))
            ext = float(np.degrees(np.arccos(cosang)))
        except Exception:
            ext = None
        angles.append(ext)

    cleaned = [(0.0 if a is None else a) for a in angles]
    smoothed = _smooth(cleaned)

    idx = int(np.argmax(smoothed))
    conf = float(max(0.0, min(1.0, pose_frames[idx].confidence))) * 100.0
    return EventFrame(frame=idx, conf=conf)


# -----------------------------------------------------
# C-2 UAH detection (flexion + elevation + rotation)
# -----------------------------------------------------
def detect_uah_c2(pose_frames, mapper, rel_idx) -> Optional[EventFrame]:
    if rel_idx <= 2:
        return None

    flex = _flexion_list(pose_frames, mapper)
    elev = _elevation_list(pose_frames, mapper)
    rot = _rotation_list(pose_frames, mapper)

    # --- A) Flexion minimum baseline ---
    flex_idx = int(np.argmin(flex[:rel_idx]))

    # --- B) Elevation rise window ---
    elev_segment = elev[:rel_idx]
    elev_vel = np.gradient(elev_segment)
    accel_candidates = [i for i, v in enumerate(elev_vel) if v > 0.01]

    if accel_candidates:
        elev_idx = int(np.median(accel_candidates))
    else:
        elev_idx = flex_idx

    # --- C) Shoulder rotation (torso uncoiling begins) ---
    rot_vel = np.gradient(rot[:rel_idx])
    rot_candidates = [i for i, v in enumerate(rot_vel) if v > 0.01]

    if rot_candidates:
        rot_idx = int(np.median(rot_candidates))
    else:
        rot_idx = flex_idx

    # --- Combine (C-2 logic) ---
    raw_idx = int(np.median([flex_idx, elev_idx, rot_idx]))

    # Bounded correction window (±8 frames from flex minimum)
    uah_idx = int(np.clip(raw_idx, flex_idx - 8, flex_idx + 8))

    conf = float(max(0.0, min(1.0, pose_frames[uah_idx].confidence))) * 100.0
    return EventFrame(frame=uah_idx, conf=conf)


# -----------------------------------------------------
# FFC (same as before)
# -----------------------------------------------------
def detect_ffc(pose_frames, mapper, rel_idx) -> Optional[EventFrame]:
    Ys = []
    ankle = mapper.primary["ankle"]

    for i in range(rel_idx):
        lm = pose_frames[i].landmarks
        Ys.append(1.0 if lm is None else float(lm[ankle]["y"]))

    cleaned = _smooth(Ys)
    idx = int(np.argmin(cleaned))
    conf = float(max(0.0, min(1.0, pose_frames[idx].confidence))) * 100.0
    return EventFrame(frame=idx, conf=conf)


# -----------------------------------------------------
# BFC (same as before)
# -----------------------------------------------------
def detect_bfc(pose_frames, mapper, ffc_idx) -> Optional[EventFrame]:
    if ffc_idx <= 1:
        return None

    Ys = []
    ankle = mapper.primary["ankle"]

    for i in range(ffc_idx):
        lm = pose_frames[i].landmarks
        Ys.append(1.0 if lm is None else float(lm[ankle]["y"]))

    cleaned = _smooth(Ys)
    idx = int(np.argmin(cleaned))
    conf = float(max(0.0, min(1.0, pose_frames[idx].confidence))) * 100.0
    return EventFrame(frame=idx, conf=conf)


# -----------------------------------------------------
# MAIN EVENTS STAGE
# -----------------------------------------------------
def run(ctx: Context) -> Context:
    try:
        pose_frames = ctx.pose.frames
        if not pose_frames or len(pose_frames) < 5:
            ctx.events.error = "Insufficient pose frames"
            return ctx

        mapper = LandmarkMapper(ctx.input.hand)

        # Apply visibility filter
        filtered = [pf for pf in pose_frames if _is_valid_frame(pf, mapper)]
        if len(filtered) >= 5:
            pose_frames = filtered

        # 1. Release
        release = detect_release(pose_frames, mapper)
        if release is None:
            ctx.events.error = "Release not found"
            return ctx

        # 2. UAH (C-2 Adaptive)
        uah = detect_uah_c2(pose_frames, mapper, release.frame)
        if uah is None:
            uah = EventFrame(frame=max(0, release.frame - 5), conf=10.0)

        # Ordering correction
        if uah.frame > release.frame:
            uah.frame = max(0, release.frame - 3)

        # 3. FFC
        ffc = detect_ffc(pose_frames, mapper, release.frame)
        if ffc is None:
            ffc = EventFrame(frame=max(0, uah.frame - 5), conf=10.0)

        # 4. BFC
        bfc = detect_bfc(pose_frames, mapper, ffc.frame)
        if bfc is None:
            bfc = EventFrame(frame=max(0, ffc.frame - 5), conf=10.0)

        ctx.events.release = release
        ctx.events.uah = uah
        ctx.events.ffc = ffc
        ctx.events.bfc = bfc
        ctx.events.error = None
        return ctx

    except Exception as e:
        ctx.events.error = str(e)
        return ctx

