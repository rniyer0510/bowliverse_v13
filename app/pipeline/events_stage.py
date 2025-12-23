# app/pipeline/events_stage.py
"""
Bowliverse v13.8 — EVENTS STAGE (C-2 Adaptive UAH + Reverse FFC)
--------------------------------------------------------------

Design rules:
- Canonical order (reverse-anchored): BFC → FFC → UAH → Release
- Release is the most reliable anchor
- UAH refined using flexion + elevation + rotation (C-2)
- FFC detected in REVERSE, anchored at UAH
- BFC derived conservatively in reverse (placeholder; refined later)
- Unified analysis window is constructed ONCE and passed downstream
- No executable logic at module scope
"""

from typing import Optional
import numpy as np

from app.models.context import Context
from app.models.events_model import EventFrame
from app.utils.landmarks import LandmarkMapper
from app.pipeline.ffc_reverse import detect_ffc_reverse


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
        return sh >= min_vis and el >= min_vis and wr >= min_vis
    except Exception:
        return False


# -----------------------------------------------------
# Flexion curve (internal)
# -----------------------------------------------------
def _flexion_list(pose_frames, mapper):
    out = []
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            out.append(0.0)
            continue
        try:
            S = mapper.vec(lm, "shoulder")
            E = mapper.vec(lm, "elbow")
            W = mapper.vec(lm, "wrist")
            hum = S - E
            fore = W - E
            denom = (np.linalg.norm(hum) * np.linalg.norm(fore)) + 1e-9
            cosang = np.clip(np.dot(hum, fore) / denom, -1.0, 1.0)
            ext = float(np.degrees(np.arccos(cosang)))
            out.append(180.0 - ext)
        except Exception:
            out.append(0.0)
    return _smooth(out)


# -----------------------------------------------------
# Elevation (shoulder above elbow)
# -----------------------------------------------------
def _elevation_list(pose_frames, mapper):
    sh = mapper.primary["shoulder"]
    el = mapper.primary["elbow"]
    vals = []
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            vals.append(0.0)
            continue
        vals.append(float(lm[sh]["y"] - lm[el]["y"]))
    return _smooth(vals)


# -----------------------------------------------------
# Shoulder rotation velocity
# -----------------------------------------------------
def _rotation_list(pose_frames, mapper):
    sh = mapper.primary["shoulder"]
    hip = mapper.primary["hip"]
    vals = []
    prev = None
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            vals.append(0.0)
            continue
        cur = lm[sh]["x"] - lm[hip]["x"]
        vals.append(0.0 if prev is None else cur - prev)
        prev = cur
    return _smooth(vals)


# -----------------------------------------------------
# Release detection
# -----------------------------------------------------
def detect_release(pose_frames, mapper) -> Optional[EventFrame]:
    angles = []
    for pf in pose_frames:
        lm = pf.landmarks
        if lm is None:
            angles.append(0.0)
            continue
        try:
            S = mapper.vec(lm, "shoulder")
            E = mapper.vec(lm, "elbow")
            W = mapper.vec(lm, "wrist")
            hum = S - E
            fore = W - E
            denom = (np.linalg.norm(hum) * np.linalg.norm(fore)) + 1e-9
            cosang = np.clip(np.dot(hum, fore) / denom, -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cosang))))
        except Exception:
            angles.append(0.0)

    idx = int(np.argmax(_smooth(angles)))
    conf = float(max(0.0, min(1.0, pose_frames[idx].confidence))) * 100.0
    return EventFrame(frame=idx, conf=conf)


# -----------------------------------------------------
# UAH detection (C-2)
# -----------------------------------------------------
def detect_uah_c2(pose_frames, mapper, rel_idx) -> Optional[EventFrame]:
    if rel_idx <= 3:
        return None

    flex = _flexion_list(pose_frames, mapper)
    elev = _elevation_list(pose_frames, mapper)
    rot = _rotation_list(pose_frames, mapper)

    flex_idx = int(np.argmin(flex[:rel_idx]))

    elev_vel = np.gradient(elev[:rel_idx])
    rot_vel = np.gradient(rot[:rel_idx])

    elev_idx = (
        int(np.median([i for i, v in enumerate(elev_vel) if v > 0.01]))
        if any(v > 0.01 for v in elev_vel)
        else flex_idx
    )

    rot_idx = (
        int(np.median([i for i, v in enumerate(rot_vel) if v > 0.01]))
        if any(v > 0.01 for v in rot_vel)
        else flex_idx
    )

    raw = int(np.median([flex_idx, elev_idx, rot_idx]))
    uah_idx = int(np.clip(raw, flex_idx - 8, flex_idx + 8))

    conf = float(max(0.0, min(1.0, pose_frames[uah_idx].confidence))) * 100.0
    return EventFrame(frame=uah_idx, conf=conf)


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

        # Visibility filter
        valid = [pf for pf in pose_frames if _is_valid_frame(pf, mapper)]
        if len(valid) >= 5:
            pose_frames = valid

        # -------------------------
        # Release (anchor)
        # -------------------------
        release = detect_release(pose_frames, mapper)
        if release is None:
            ctx.events.error = "Release not found"
            return ctx

        # -------------------------
        # UAH (reverse-bounded)
        # -------------------------
        uah = detect_uah_c2(pose_frames, mapper, release.frame)
        if uah is None:
            uah = EventFrame(frame=max(0, release.frame - 6), conf=10.0)

        if uah.frame >= release.frame:
            uah.frame = max(0, release.frame - 3)

        # -------------------------
        # FFC — reverse (anchored at UAH)
        # -------------------------
        ffc = detect_ffc_reverse(
            pose_frames=pose_frames,
            anchor_idx=uah.frame,
            hand=ctx.input.hand,
        )

        if ffc is None:
            ffc = EventFrame(frame=max(0, uah.frame - 6), conf=10.0)

        # -------------------------
        # BFC — minimal reverse placeholder
        # -------------------------
        bfc = None
        if ffc and ffc.frame > 3:
            bfc = EventFrame(
                frame=max(0, ffc.frame - 8),
                conf=10.0
            )

        # -------------------------
        # Unified analysis window
        # -------------------------
        analysis_window = {
            "bfc": bfc.frame if bfc else None,
            "ffc": ffc.frame if ffc else None,
            "uah": uah.frame,
            "release": release.frame,
        }

        # Hard invariant — NEVER relax
        if None not in analysis_window.values():
            assert (
                analysis_window["bfc"]
                < analysis_window["ffc"]
                < analysis_window["uah"]
                < analysis_window["release"]
            ), f"Invalid reverse window ordering: {analysis_window}"

        # -------------------------
        # Commit to context
        # -------------------------
        ctx.events.release = release
        ctx.events.uah = uah
        ctx.events.ffc = ffc
        ctx.events.bfc = bfc
        ctx.events.window = analysis_window
        ctx.events.error = None

        return ctx

    except Exception as e:
        ctx.events.error = str(e)
        return ctx

