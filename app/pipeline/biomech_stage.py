# app/pipeline/biomech_stage.py
import numpy as np
from app.models.context import Context
from app.models.biomech_model import BiomechElbowModel, ReleaseHeightModel
from app.utils.landmarks import LandmarkMapper
from app.utils.angles import elbow_flexion
from app.utils.logger import log

# NEW primitives
from app.pipeline.backfoot_contact import run as backfoot_contact_run
from app.pipeline.hip_alignment import run as hip_alignment_run
from app.pipeline.shoulder_alignment import run as shoulder_alignment_run
from app.pipeline.shoulder_hip_separation import run as shoulder_hip_separation_run


MAX_INTERP_GAP = 7   # D2 rule


# ---------------------------------------------------------
# Utility: linear interpolation between two joints
# ---------------------------------------------------------
def interp_vec(a, b, t):
    return a + t * (b - a)


# ---------------------------------------------------------
# Utility: interpolate shoulder/elbow/wrist only
# ---------------------------------------------------------
def interpolate_arm_joints(pose_frames, mapper, start_idx, end_idx):
    """
    Only interpolate shoulder, elbow, wrist.
    If more than MAX_INTERP_GAP frames are missing → abort.
    """

    missing = []
    for i in range(start_idx, end_idx + 1):
        if pose_frames[i].landmarks is None:
            missing.append(i)

    if not missing:
        return True  # nothing to repair

    # If too many missing frames → do not interpolate
    if len(missing) > MAX_INTERP_GAP:
        log(f"[Interp] Too many missing frames ({len(missing)}). Aborting interpolation.")
        return False

    # For each missing frame, interpolate arm joints only
    for idx in missing:
        # Find previous valid frame
        prev_idx = idx - 1
        while prev_idx >= start_idx and pose_frames[prev_idx].landmarks is None:
            prev_idx -= 1

        # Find next valid frame
        next_idx = idx + 1
        while next_idx <= end_idx and pose_frames[next_idx].landmarks is None:
            next_idx += 1

        # If no bounding valid frames → cannot interpolate
        if prev_idx < start_idx or next_idx > end_idx:
            log(f"[Interp] Frame {idx} cannot be bounded. Skipping.")
            continue

        # Extract vectors
        prev_lm = pose_frames[prev_idx].landmarks
        next_lm = pose_frames[next_idx].landmarks

        # Prepare new landmark list (copy of previous)
        new_lm = [None] * len(prev_lm)

        # Retrieve joint indices
        sh = mapper.primary["shoulder"]
        el = mapper.primary["elbow"]
        wr = mapper.primary["wrist"]

        # For only the 3 arm joints → interpolate
        for joint in [sh, el, wr]:
            pA = np.array([
                prev_lm[joint]["x"],
                prev_lm[joint]["y"],
                prev_lm[joint]["z"]
            ], float)

            pB = np.array([
                next_lm[joint]["x"],
                next_lm[joint]["y"],
                next_lm[joint]["z"]
            ], float)

            t = (idx - prev_idx) / (next_idx - prev_idx)
            pI = interp_vec(pA, pB, t)

            new_lm[joint] = {
                "x": float(pI[0]),
                "y": float(pI[1]),
                "z": float(pI[2]),
                "vis": 1.0  # interpolated confidence
            }

        # Fill the remaining joints from prev_lm unchanged
        for j in range(len(prev_lm)):
            if new_lm[j] is None:
                new_lm[j] = prev_lm[j]

        # Assign repaired frame
        pose_frames[idx].landmarks = new_lm
        pose_frames[idx].confidence = 1.0

        log(f"[Interp] Frame {idx} repaired via interpolation.")

    return True


# ---------------------------------------------------------
# Main biomech stage
# ---------------------------------------------------------
def run(ctx: Context) -> Context:
    try:
        log("[INFO] BiomechStage: Starting biomechanical analysis")

        pose_frames = ctx.pose.frames
        if not pose_frames:
            ctx.biomech.error = "Missing pose frames"
            return ctx

        events = ctx.events
        if not events.release or not events.uah:
            ctx.biomech.error = "Missing key event frames"
            return ctx

        # -------------------------------------------------------------
        # NEW: Alignment primitives (matrix-ready)
        # -------------------------------------------------------------
        backfoot_contact_run(ctx)
        hip_alignment_run(ctx)
        shoulder_alignment_run(ctx)
        shoulder_hip_separation_run(ctx)

        f_rel = events.release.frame
        f_uah = events.uah.frame

        if not (0 <= f_uah < len(pose_frames)) or not (0 <= f_rel < len(pose_frames)):
            ctx.biomech.error = "Invalid event frames"
            return ctx

        log(f"[DEBUG] BiomechStage: f_rel={f_rel}, f_uah={f_uah}, total_frames={len(pose_frames)}")
        log(f"[DEBUG] BiomechStage: Handedness={ctx.input.hand}")

        mapper = LandmarkMapper(ctx.input.hand)

        # -------------------------------------------------------------
        # C2 + D2: interpolate missing ARM joints between UAH → Release
        # -------------------------------------------------------------
        ok = interpolate_arm_joints(pose_frames, mapper, f_uah, f_rel)
        if not ok:
            ctx.biomech.error = "Insufficient valid joint frames"
            return ctx

        # -------------------------------------------------------------
        # Build internal flexion curve
        # -------------------------------------------------------------
        flex_list = []
        for i, pf in enumerate(pose_frames[:f_rel + 1]):
            if pf.landmarks is None:
                flex_list.append(0.0)
                continue

            S = mapper.vec(pf.landmarks, "shoulder")
            E = mapper.vec(pf.landmarks, "elbow")
            W = mapper.vec(pf.landmarks, "wrist")

            ang = float(elbow_flexion(S, E, W))
            flex_list.append(ang)

            if i % 10 == 0:
                log(f"[DEBUG] [ElbowTrace] Frame={i} InternalFlex={ang:.2f}")

        if len(flex_list) < 3:
            ctx.biomech.error = "Insufficient valid flexion curve"
            return ctx

        # Median-smoothed values
        def median_window(values, idx, radius=2):
            lo = max(0, idx - radius)
            hi = min(len(values) - 1, idx + radius)
            return float(np.median(values[lo:hi+1]))

        uah_int = median_window(flex_list, f_uah)
        rel_int = median_window(flex_list, f_rel)

        uah_ext = 180.0 - uah_int
        rel_ext = 180.0 - rel_int

        extension_icc = rel_ext - uah_ext
        if extension_icc < 0:
            extension_icc = 0.0

        peak_internal = float(max(flex_list))
        peak_external = 180.0 - peak_internal

        log(f"[DEBUG] UAH internal={uah_int:.2f}, Release internal={rel_int:.2f}")
        log(f"[DEBUG] UAH external={uah_ext:.2f}, Release external={rel_ext:.2f}")
        log(f"[DEBUG] ICC extension={extension_icc:.2f} deg")
        log(f"[DEBUG] Peak internal={peak_internal:.2f}, Peak external={peak_external:.2f}")

        # -------------------------------------------------------------
        # Release height
        # -------------------------------------------------------------
        pf_rel = pose_frames[f_rel]
        W_rel = mapper.vec(pf_rel.landmarks, "wrist")
        E_rel = mapper.vec(pf_rel.landmarks, "elbow")
        S_rel = mapper.vec(pf_rel.landmarks, "shoulder")

        wrist_y = float(W_rel[1])
        torso_y = float((S_rel[1] + E_rel[1]) / 2.0)
        norm_height = wrist_y - torso_y

        log(f"[DEBUG] Release Height: WristY={wrist_y:.4f}, NormHeight={norm_height:.4f}")

        # -------------------------------------------------------------
        # Confidence scores
        # -------------------------------------------------------------
        elbow_conf = (events.release.conf + events.uah.conf) / 2.0
        release_height_conf = float(events.release.conf)

        # -------------------------------------------------------------
        # Assign biomech models
        # -------------------------------------------------------------
        ctx.biomech.elbow = BiomechElbowModel(
            uah_angle=uah_ext,
            release_angle=rel_ext,
            peak_extension_angle_deg=peak_external,
            peak_extension_frame=f_rel,
            extension_deg=extension_icc,
            extension_raw_deg=peak_external,
            extension_error_margin_deg=6.0,
            extension_note="Phase-1 Stable Reading (Interp Enabled)",
        )

        ctx.biomech.release_height = ReleaseHeightModel(
            norm_height=norm_height,
            wrist_y=wrist_y,
        )

        ctx.biomech.elbow_conf = elbow_conf
        ctx.biomech.release_height_conf = release_height_conf
        ctx.biomech.angle_plane = {
            "uah_plane": "projected",
            "release_plane": "projected",
        }

        log("[INFO] BiomechStage: Completed successfully")
        ctx.biomech.error = None
        return ctx

    except Exception as e:
        ctx.biomech.error = str(e)
        log(f"[ERROR] BiomechStage Exception: {e}")
        return ctx
