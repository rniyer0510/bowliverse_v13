# app/pipeline/biomech_stage.py
"""
Bowliverse v13.7 — BIOMECH STAGE
ICC-accurate flexion → excursion model
with Baseball-style continuous flexion curve smoothing
and Confidence-Weighted Extension (CWE) for real-world videos.

This solves:
- Inflated extension from phone angles
- YouTube distortions
- McGrath-like perfect actions showing 90°+
- Amateur low-visibility clips
"""

from typing import List
import numpy as np

from app.pipeline.context import Context
from app.utils.angles import elbow_flexion
from app.utils.landmarks import LandmarkMapper
from app.models.biomech_model import BiomechElbowModel, BiomechReleaseHeightModel


# -----------------------------
# Gaussian smoothing
# -----------------------------
def gaussian_smooth(values: List[float], sigma=1.0) -> List[float]:
    if len(values) < 3:
        return values
    k = int(3 * sigma)
    kernel = np.exp(-np.linspace(-k, k, 2 * k + 1) ** 2 / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same").tolist()


# -----------------------------
# Visibility confidence
# -----------------------------
def _visibility(mapper: LandmarkMapper, lm):
    pts = [mapper.primary["shoulder"], mapper.primary["elbow"], mapper.primary["wrist"]]
    vis = []
    for p in pts:
        if p in lm and lm[p]["visibility"] is not None:
            vis.append(float(lm[p]["visibility"]))
    if not vis:
        return 0.0
    return float(sum(vis) / len(vis))


# -----------------------------
# MAIN
# -----------------------------
def run(ctx: Context) -> Context:
    events = ctx.events
    frames = ctx.pose.frames

    # -----------------------------------------------------
    # Required event frames
    # -----------------------------------------------------
    if events.release is None or events.uah is None:
        ctx.biomech.error = "Missing UAH or Release frame."
        return ctx

    rel_idx = events.release.frame
    uah_idx = events.uah.frame

    if rel_idx >= len(frames) or uah_idx >= len(frames):
        ctx.biomech.error = "Event frame index out of bounds."
        return ctx

    # Ensure UAH → Release ordering
    if uah_idx > rel_idx:
        uah_idx, rel_idx = rel_idx, uah_idx

    mapper = LandmarkMapper(ctx.input.hand)

    # -----------------------------------------------------
    # Build continuous flexion curve (Baseball method)
    # -----------------------------------------------------
    flexions = []
    for idx in range(uah_idx, rel_idx + 1):
        lm = frames[idx].landmarks

        S = mapper._safe_vec(lm, mapper.primary["shoulder"])
        E = mapper._safe_vec(lm, mapper.primary["elbow"])
        W = mapper._safe_vec(lm, mapper.primary["wrist"])

        flexions.append(elbow_flexion(S, E, W))

    # Smooth flexion curve
    flex_smooth = gaussian_smooth(flexions, sigma=1.0)

    # ICC excursion = max - min
    flex_min = float(min(flex_smooth))
    flex_max = float(max(flex_smooth))
    excursion = float(max(0.0, flex_max - flex_min))

    uah_flex = float(flex_smooth[0])
    rel_flex = float(flex_smooth[-1])

    # -----------------------------------------------------
    # Release height
    # -----------------------------------------------------
    rel_frame = frames[rel_idx]
    LS, RS = mapper.shoulders_pair(rel_frame.landmarks)
    LH, RH = mapper.hips_pair(rel_frame.landmarks)

    shoulder_mid = (LS + RS) / 2.0
    hip_mid = (LH + RH) / 2.0
    torso_len = abs(shoulder_mid[1] - hip_mid[1]) + 1e-6

    wrist_y = float(mapper._safe_vec(rel_frame.landmarks, mapper.primary["wrist"])[1])
    norm_height = float((shoulder_mid[1] - wrist_y) / torso_len)

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------
    vis_rel = _visibility(mapper, rel_frame.landmarks)
    vis_uah = _visibility(mapper, frames[uah_idx].landmarks)

    elbow_conf = int(
        50 * ((rel_frame.confidence + frames[uah_idx].confidence) / 2.0)
        + 50 * ((vis_rel + vis_uah) / 2.0)
    )
    elbow_conf = max(0, min(elbow_conf, 100))

    release_height_conf = int(100 * rel_frame.confidence)
    release_height_conf = max(0, min(release_height_conf, 100))

    if elbow_conf > 85:
        note = "High confidence estimate"
    elif elbow_conf > 60:
        note = "Moderate confidence; minor occlusion"
    else:
        note = "Low confidence; visibility issues"

    # -----------------------------------------------------
    # CONFIDENCE-WEIGHTED EXTENSION (CWE)
    # -----------------------------------------------------
    extension_raw = excursion
    conf_factor = elbow_conf / 100.0

    extension_final = extension_raw

    # Dampening for off-angle / low-confidence amateur footage
    if extension_raw > 40 and conf_factor < 0.85:
        extension_final = extension_raw * conf_factor * 0.75

    # Never exceed raw
    extension_final = min(extension_final, extension_raw)

    # -----------------------------------------------------
    # Save to context
    # -----------------------------------------------------
    ctx.biomech.elbow = BiomechElbowModel(
        uah_angle=uah_flex,
        release_angle=rel_flex,
        extension_deg=extension_final,
        extension_raw_deg=extension_raw,
        extension_error_margin_deg=6.0,
        extension_note=note,
    )

    ctx.biomech.release_height = BiomechReleaseHeightModel(
        norm_height=norm_height,
        wrist_y=wrist_y,
    )

    ctx.biomech.elbow_conf = elbow_conf
    ctx.biomech.release_height_conf = release_height_conf
    ctx.biomech.angle_plane = {"uah_plane": "projected", "release_plane": "projected"}

    ctx.biomech.error = None
    return ctx

