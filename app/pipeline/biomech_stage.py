# app/pipeline/biomech_stage.py

import numpy as np
from app.models.context import Context
from app.models.biomech_model import BiomechElbowModel, ReleaseHeightModel
from app.utils.landmarks import LandmarkMapper
from app.utils.angles import elbow_flexion, compute_pair_angle_series
from app.utils.logger import log

# Alignment primitives
from app.pipeline.hip_alignment import run as hip_alignment_run
from app.pipeline.shoulder_alignment import run as shoulder_alignment_run
from app.pipeline.shoulder_hip_separation import run as shoulder_hip_separation_run
from app.pipeline.backfoot_landing_angle import run as backfoot_landing_run

# Risk calculators
from app.risk.hip_shoulder_mismatch_risk import HipShoulderMismatchRisk
from app.risk.lateral_trunk_lean import LateralTrunkLeanRisk
from app.risk.front_foot_braking import FrontFootBrakingRisk
from app.risk.front_knee_collapse import FrontKneeCollapseRisk
from app.risk.formatter import format_risk
from app.risk.aggregator import aggregate_risks

MAX_INTERP_GAP = 7


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def interp_vec(a, b, t):
    return a + t * (b - a)


def interpolate_arm_joints(pose_frames, mapper, start_idx, end_idx):
    missing = [i for i in range(start_idx, end_idx + 1) if pose_frames[i].landmarks is None]

    if not missing:
        return True

    if len(missing) > MAX_INTERP_GAP:
        log(f"[Interp] Too many missing frames ({len(missing)}).")
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

        for joint in (
            mapper.primary["shoulder"],
            mapper.primary["elbow"],
            mapper.primary["wrist"],
        ):
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


# ------------------------------------------------------------
# Main Stage
# ------------------------------------------------------------
def run(ctx: Context) -> Context:
    try:
        log("[INFO] BiomechStage: Starting")

        pose_frames = ctx.pose.frames
        events = ctx.events

        if not pose_frames or not events.release or not events.uah:
            ctx.biomech.error = "Missing pose or mandatory events"
            return ctx

        f_rel = events.release.frame
        f_uah = events.uah.frame

        mapper = LandmarkMapper(ctx.input.hand)

        if not interpolate_arm_joints(pose_frames, mapper, f_uah, f_rel):
            ctx.biomech.error = "Arm landmarks insufficient"
            return ctx

        # ------------------------------------------------------------
        # Elbow mechanics
        # ------------------------------------------------------------
        flex = []
        for pf in pose_frames[: f_rel + 1]:
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

        uah_ext = 180.0 - median(flex, f_uah)
        rel_ext = 180.0 - median(flex, f_rel)
        extension = max(0.0, rel_ext - uah_ext)
        peak_external = 180.0 - max(flex)

        # ------------------------------------------------------------
        # Release height
        # ------------------------------------------------------------
        pf = pose_frames[f_rel]
        W = mapper.vec(pf.landmarks, "wrist")
        S = mapper.vec(pf.landmarks, "shoulder")
        E = mapper.vec(pf.landmarks, "elbow")
        norm_height = float(W[1] - (S[1] + E[1]) / 2.0)

        # ------------------------------------------------------------
        # Alignment primitives
        # ------------------------------------------------------------
        hip_alignment_run(ctx)
        shoulder_alignment_run(ctx)
        shoulder_hip_separation_run(ctx)
        backfoot_landing_run(ctx)

        # ------------------------------------------------------------
        # Store biomech outputs
        # ------------------------------------------------------------
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

        ctx.biomech.elbow_conf = float((events.release.conf + events.uah.conf) / 2.0)
        ctx.biomech.release_height_conf = float(events.release.conf)

        # ------------------------------------------------------------
        # RISK COMPUTATION (practical, gated)
        # ------------------------------------------------------------
        formatted = []

        ffb = None
        hsm = None

        # Front-Foot Braking (primary load indicator)
        try:
            ffb = FrontFootBrakingRisk().compute(
                pose_frames=pose_frames,
                mapper=mapper,
                ffc=events.ffc.frame,
                release=events.release.frame,
                fps=ctx.pose.fps,
                I=0.7,
            )
            if ffb:
                formatted.append(format_risk(ffb))
        except Exception as e:
            log(f"[WARN] FFB skipped: {e}")

        # Hip–Shoulder Mismatch (only if data supports it)
        try:
            f_bfc = events.bfc.frame
            f_ffc = events.ffc.frame

            hip = compute_pair_angle_series(
                pose_frames, mapper, f_bfc, f_ffc, mapper.hips_pair
            )
            sh = compute_pair_angle_series(
                pose_frames, mapper, f_bfc, f_ffc, mapper.shoulders_pair
            )

            hsm = HipShoulderMismatchRisk().compute(
                hip_angles=hip,
                shoulder_angles=sh,
                events={"BFC": f_bfc, "FFC": f_ffc},
                dt=1.0 / ctx.pose.fps,
                I=0.7,
            )
            if hsm:
                formatted.append(format_risk(hsm))
        except Exception as e:
            log(f"[WARN] HSM skipped: {e}")

        # Lateral Trunk Lean (cannot dominate alone)
        try:
            ltl = LateralTrunkLeanRisk().compute(
                pose_frames=pose_frames,
                mapper=mapper,
                ffc=events.ffc.frame,
                uah=f_uah,
                I=0.7,
            )

            if ltl:
                supporting = False
                for r in (ffb, hsm):
                    if r and isinstance(r, dict) and r.get("level") in ("MEDIUM", "HIGH"):
                        supporting = True
                        break

                if supporting:
                    formatted.append(format_risk(ltl))
                else:
                    downgraded = dict(ltl)
                    downgraded["level"] = "LOW"
                    formatted.append(format_risk(downgraded))

        except Exception as e:
            log(f"[WARN] LTL skipped: {e}")

        # Front-Knee Collapse (structural stability)
        try:
            ffb_level = ffb.get("level") if isinstance(ffb, dict) else None

            fnc = FrontKneeCollapseRisk().compute(
                pose_frames=pose_frames,
                mapper=mapper,
                ffc=events.ffc.frame,
                release=events.release.frame,
                fps=ctx.pose.fps,
                I=0.7,
            )

            if fnc:
                formatted.append(format_risk(fnc))

        except Exception as e:
            log(f"[WARN] FNC skipped: {e}")

        # ------------------------------------------------------------
        # Final aggregation
        # ------------------------------------------------------------
        ctx.biomech.risk = {
            "overall": aggregate_risks(formatted),
            "breakdown": formatted,
        }

        ctx.biomech.error = None
        log("[INFO] BiomechStage: Completed")
        return ctx

    except Exception as e:
        ctx.biomech.error = str(e)
        log(f"[ERROR] BiomechStage: {e}")
        return ctx


        # ------------------------------------------------------------
        # CUES (interpretation layer — Phase-1)
        # ------------------------------------------------------------
        try:
            from app.cues.cue_engine import build_cues
            ctx.cues = build_cues(ctx)
        except Exception as e:
            log(f"[WARN] Cue engine skipped: {e}")
            ctx.cues = {"list": []}
