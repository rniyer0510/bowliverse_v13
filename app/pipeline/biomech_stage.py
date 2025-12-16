# app/pipeline/biomech_stage.py

import numpy as np
from app.models.context import Context
from app.models.biomech_model import BiomechElbowModel, ReleaseHeightModel
from app.utils.landmarks import LandmarkMapper
from app.utils.angles import elbow_flexion
from app.utils.logger import log

# Alignment primitives (camera-agnostic)
from app.pipeline.hip_alignment import run as hip_alignment_run
from app.pipeline.shoulder_alignment import run as shoulder_alignment_run
from app.pipeline.shoulder_hip_separation import run as shoulder_hip_separation_run
from app.pipeline.backfoot_landing_angle import run as backfoot_landing_run


MAX_INTERP_GAP = 7   # D2 rule


def interp_vec(a, b, t):
    return a + t * (b - a)


def interpolate_arm_joints(pose_frames, mapper, start_idx, end_idx):
    missing = []
    for i in range(start_idx, end_idx + 1):
        if pose_frames[i].landmarks is None:
            missing.append(i)

    if not missing:
        return True

    if len(missing) > MAX_INTERP_GAP:
        log(f"[Interp] Too many missing frames ({len(missing)}). Aborting.")
        return False

    for idx in missing:
        prev_idx = idx - 1
        while prev_idx >= start_idx and pose_frames[prev_idx].landmarks is None:
            prev_idx -= 1

        next_idx = idx + 1
        while next_idx <= end_idx and pose_frames[next_idx].landmarks is None:
            next_idx += 1

        if prev_idx < start_idx or next_idx > end_idx:
            continue

        prev_lm = pose_frames[prev_idx].landmarks
        next_lm = pose_frames[next_idx].landmarks

        new_lm = [None] * len(prev_lm)

        sh = mapper.primary["shoulder"]
        el = mapper.primary["elbow"]
        wr = mapper.primary["wrist"]

        for joint in [sh, el, wr]:
            pA = np.array([prev_lm[joint]["x"], prev_lm[joint]["y"], prev_lm[joint]["z"]])
            pB = np.array([next_lm[joint]["x"], next_lm[joint]["y"], next_lm[joint]["z"]])
            t = (idx - prev_idx) / (next_idx - prev_idx)
            pI = interp_vec(pA, pB, t)

            new_lm[joint] = {
                "x": float(pI[0]),
                "y": float(pI[1]),
                "z": float(pI[2]),
                "vis": 1.0,
            }

        for j in range(len(prev_lm)):
            if new_lm[j] is None:
                new_lm[j] = prev_lm[j]

        pose_frames[idx].landmarks = new_lm
        pose_frames[idx].confidence = 1.0

    return True


def run(ctx: Context) -> Context:
    try:
        log("[INFO] BiomechStage: Starting")

        pose_frames = ctx.pose.frames
        events = ctx.events

        if not pose_frames:
            ctx.biomech.error = "Missing pose frames"
            return ctx

        if not events.release or not events.uah:
            ctx.biomech.error = "Missing mandatory events (Release / UAH)"
            return ctx

        f_rel = events.release.frame
        f_uah = events.uah.frame

        if not (0 <= f_uah < len(pose_frames)) or not (0 <= f_rel < len(pose_frames)):
            ctx.biomech.error = "Invalid event frame indices"
            return ctx

        mapper = LandmarkMapper(ctx.input.hand)

        if not interpolate_arm_joints(pose_frames, mapper, f_uah, f_rel):
            ctx.biomech.error = "Insufficient arm visibility for interpolation"
            return ctx

        flex = []
        for pf in pose_frames[:f_rel + 1]:
            if pf.landmarks is None:
                flex.append(0.0)
                continue
            S = mapper.vec(pf.landmarks, "shoulder")
            E = mapper.vec(pf.landmarks, "elbow")
            W = mapper.vec(pf.landmarks, "wrist")
            flex.append(float(elbow_flexion(S, E, W)))

        def median(vals, idx, r=2):
            lo = max(0, idx - r)
            hi = min(len(vals), idx + r + 1)
            return float(np.median(vals[lo:hi]))

        uah_int = median(flex, f_uah)
        rel_int = median(flex, f_rel)

        uah_ext = 180.0 - uah_int
        rel_ext = 180.0 - rel_int
        extension = max(0.0, rel_ext - uah_ext)

        peak_internal = max(flex)
        peak_external = 180.0 - peak_internal

        pf = pose_frames[f_rel]
        W = mapper.vec(pf.landmarks, "wrist")
        S = mapper.vec(pf.landmarks, "shoulder")
        E = mapper.vec(pf.landmarks, "elbow")

        norm_height = float(W[1] - (S[1] + E[1]) / 2.0)

        hip_alignment_run(ctx)
        shoulder_alignment_run(ctx)
        shoulder_hip_separation_run(ctx)
        backfoot_landing_run(ctx)

        # ✅ Correct model → dict conversion
        ctx.biomech.elbow = BiomechElbowModel(
            uah_angle=uah_ext,
            release_angle=rel_ext,
            peak_extension_angle_deg=peak_external,
            peak_extension_frame=f_rel,
            extension_deg=extension,
            extension_raw_deg=peak_external,
            extension_error_margin_deg=6.0,
            extension_note="Stable (Interp Enabled)",
        ).model_dump()

        ctx.biomech.release_height = ReleaseHeightModel(
            norm_height=norm_height,
            wrist_y=float(W[1]),
        ).model_dump()

        ctx.biomech.elbow_conf = float(
            (events.release.conf + events.uah.conf) / 2.0
        )
        ctx.biomech.release_height_conf = float(events.release.conf)

        ctx.biomech.angle_plane = {
            "uah_plane": "projected",
            "release_plane": "projected",
        }

        ctx.biomech.error = None
        log("[INFO] BiomechStage: Completed")
        return ctx

    except Exception as e:
        ctx.biomech.error = str(e)
        log(f"[ERROR] BiomechStage: {e}")
        return ctx

