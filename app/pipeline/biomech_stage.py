# app/pipeline/biomech_stage.py

import numpy as np
from statistics import stdev
from app.models.context import Context
from app.models.biomech_model import BiomechElbowModel, ReleaseHeightModel
from app.utils.landmarks import LandmarkMapper
from app.utils.angles import elbow_flexion
from app.utils.logger import log

# Alignment primitives
from app.pipeline.hip_alignment import run as hip_alignment_run
from app.pipeline.shoulder_alignment import run as shoulder_alignment_run
from app.pipeline.shoulder_hip_separation import run as shoulder_hip_separation_run
from app.pipeline.backfoot_landing_angle import run as backfoot_landing_run

# 🔹 Decision stage (NEW – explicit wiring)
from app.pipeline.decision_stage import run as decision_run


# -----------------------------
# Tunable (but NOT hard-coded)
# -----------------------------
MAX_INTERP_GAP = 7
VIS_THRESHOLD = 0.40
ANGLE_JUMP_LIMIT = 25.0


def run(ctx: Context) -> Context:
    try:
        log("[INFO] BiomechStage: Starting")

        pose_frames = ctx.pose.frames
        events = ctx.events

        if not pose_frames or not events.release or not events.uah:
            ctx.biomech.error = "Missing pose or mandatory events"
            return ctx

        fps = ctx.pose.fps or 25.0
        f_rel = events.release.frame
        f_uah = events.uah.frame

        mapper = LandmarkMapper(ctx.input.hand)

        # ------------------------------------------------------------
        # Baseball-style elbow window definition
        # ------------------------------------------------------------
        pre_frames = max(3, int(0.08 * fps))
        post_frames = max(2, int(0.05 * fps))

        start = max(0, f_uah - pre_frames)
        end = min(len(pose_frames) - 1, f_rel + post_frames)

        angles = []
        prev_angle = None

        for idx in range(start, end + 1):
            pf = pose_frames[idx]
            if pf.landmarks is None:
                continue

            try:
                # Visibility gating
                if (
                    pf.landmarks[mapper.primary["shoulder"]]["vis"] < VIS_THRESHOLD
                    or pf.landmarks[mapper.primary["elbow"]]["vis"] < VIS_THRESHOLD
                    or pf.landmarks[mapper.primary["wrist"]]["vis"] < VIS_THRESHOLD
                ):
                    continue

                S = mapper.vec(pf.landmarks, "shoulder")
                E = mapper.vec(pf.landmarks, "elbow")
                W = mapper.vec(pf.landmarks, "wrist")

                flex = float(elbow_flexion(S, E, W))
                ext = 180.0 - flex

                if prev_angle is not None and abs(ext - prev_angle) > ANGLE_JUMP_LIMIT:
                    continue

                angles.append(ext)
                prev_angle = ext

            except Exception:
                continue

        window_frames = end - start + 1
        support_frames = len(angles)

        # ------------------------------------------------------------
        # Reliability gate
        # ------------------------------------------------------------
        min_support = max(3, int(0.35 * window_frames))
        if support_frames < min_support:
            ctx.biomech.elbow = {
                "extension_deg": None,
                "uncertainty_deg": None,
                "support_frames": support_frames,
                "window_frames": window_frames,
                "extension_note": "Insufficient reliable elbow frames",
            }
            ctx.biomech.error = None
            return ctx

        # ------------------------------------------------------------
        # Windowed excursion (robust)
        # ------------------------------------------------------------
        low = np.percentile(angles, 20)
        high = np.percentile(angles, 80)
        excursion = max(0.0, float(high - low))

        std = stdev(angles) if len(angles) > 2 else 0.0
        uncertainty = round(1.5 * std, 2)

        ctx.biomech.elbow = BiomechElbowModel(
            uah_angle=None,
            release_angle=None,
            peak_extension_angle_deg=excursion,
            peak_extension_frame=f_rel,
            extension_deg=excursion,
            extension_raw_deg=excursion,
            extension_error_margin_deg=uncertainty,
            extension_note="Windowed excursion (baseball-style)",
        ).model_dump()

        # Attach window diagnostics INSIDE elbow (safe)
        ctx.biomech.elbow["uncertainty_deg"] = uncertainty
        ctx.biomech.elbow["support_frames"] = support_frames
        ctx.biomech.elbow["window_frames"] = window_frames

        # ------------------------------------------------------------
        # Release height (unchanged logic)
        # ------------------------------------------------------------
        pf = pose_frames[f_rel]
        if pf.landmarks:
            W = mapper.vec(pf.landmarks, "wrist")
            S = mapper.vec(pf.landmarks, "shoulder")
            E = mapper.vec(pf.landmarks, "elbow")
            norm_height = float(W[1] - (S[1] + E[1]) / 2.0)
            wrist_y = float(W[1])
        else:
            norm_height = None
            wrist_y = None

        ctx.biomech.release_height = ReleaseHeightModel(
            norm_height=norm_height,
            wrist_y=wrist_y,
        ).model_dump()

        # ------------------------------------------------------------
        # Alignment primitives (windowed downstream)
        # ------------------------------------------------------------
        hip_alignment_run(ctx)
        shoulder_alignment_run(ctx)
        shoulder_hip_separation_run(ctx)
        backfoot_landing_run(ctx)

        # ------------------------------------------------------------
        # 🔹 Decision stage (ACTION CLASSIFICATION)
        # ------------------------------------------------------------
        decision_run(ctx)

        ctx.biomech.error = None
        log("[INFO] BiomechStage: Completed")
        return ctx

    except Exception as e:
        ctx.biomech.error = str(e)
        log(f"[ERROR] BiomechStage: {e}")
        return ctx

