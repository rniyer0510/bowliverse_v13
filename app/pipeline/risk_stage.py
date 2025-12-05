# app/pipeline/risk_stage.py
"""
Bowliverse v13.7 — RISK STAGE

Inputs from biomech_stage:
    elbow.extension_deg              → Confidence-weighted extension
    elbow.extension_raw_deg          → Raw excursion
    elbow_conf                       → Confidence score (0–100)
    release_height.norm_height       → Wrist relative to shoulder
    release_height_conf              → Confidence score

Outputs:
    ctx.risk.score        (0–100)
    ctx.risk.level        ("LOW_RISK", "MEDIUM_RISK", "HIGH_RISK")
    ctx.risk.details      (breakdown for UI)
"""

from app.models.context import Context

# ---------------------------------------------------------
# Thresholds — ICC-aligned & biomechanically conservative
# ---------------------------------------------------------
EXT_SAFE = 0          # <10° is very safe
EXT_MOD = 10          # 10–20° mild load
EXT_HIGH = 20         # 20–30° high load
EXT_CRIT = 30         # >30° ICC borderline
EXT_ULTRA = 40        # >40° off-angle or risky

HEIGHT_LOW = -0.25     # Wrist much below shoulder → low load
HEIGHT_NORMAL = (-0.25, 0.30)
HEIGHT_HIGH = 0.30     # Wrist significantly above shoulder


def _risk_level(score: float) -> str:
    if score < 33:
        return "LOW_RISK"
    elif score < 66:
        return "MEDIUM_RISK"
    else:
        return "HIGH_RISK"


# ---------------------------------------------------------
# MAIN STAGE
# ---------------------------------------------------------
def run(ctx: Context) -> Context:
    biomech = ctx.biomech
    elbow = biomech.elbow
    rh = biomech.release_height

    if elbow is None:
        ctx.risk.score = 0.0
        ctx.risk.level = "UNKNOWN"
        ctx.risk.details = {"reason": "Biomechanics missing"}
        return ctx

    extension = float(elbow.extension_deg)
    ext_abs = abs(extension)
    ext_conf = biomech.elbow_conf or 0

    # -----------------------------------------------------
    # Extension risk contribution (dominant in ICC models)
    # -----------------------------------------------------
    if ext_abs < EXT_MOD:  
        ext_risk = 5 * (1 - ext_conf / 100)
    elif ext_abs < EXT_HIGH:  
        ext_risk = 20 + 10 * (1 - ext_conf / 100)
    elif ext_abs < EXT_CRIT:
        ext_risk = 45 + 15 * (1 - ext_conf / 100)
    elif ext_abs < EXT_ULTRA:
        ext_risk = 65 + 20 * (1 - ext_conf / 100)
    else:
        # EXTREMELY HIGH → but could be angle-induced → dampen by confidence
        ext_risk = 85 * (1 - max(ext_conf - 50, 0) / 100)

    # -----------------------------------------------------
    # Release height contribution
    # -----------------------------------------------------
    height_risk = 0
    rh_conf = biomech.release_height_conf or 0

    if rh is not None:
        nh = float(rh.norm_height)

        if nh > HEIGHT_HIGH:
            height_risk = 20 * (1 - rh_conf / 100)
        elif nh < HEIGHT_LOW:
            height_risk = 10 * (1 - rh_conf / 100)
        else:
            height_risk = 5 * (1 - rh_conf / 100)

    # -----------------------------------------------------
    # Total risk
    # -----------------------------------------------------
    score = ext_risk + height_risk
    score = max(0.0, min(score, 100.0))

    ctx.risk.score = float(score)
    ctx.risk.level = _risk_level(score)
    ctx.risk.details = {
        "extension_deg": extension,
        "abs_extension_deg": ext_abs,
        "extension_raw_deg": elbow.extension_raw_deg,
        "extension_conf": ext_conf,
        "release_height": rh.norm_height if rh else None,
        "release_height_conf": rh_conf,
        "extension_component": ext_risk,
        "height_component": height_risk,
    }

    return ctx

