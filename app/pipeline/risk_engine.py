# app/pipeline/risk_engine.py
"""
ActionLab / Bowliverse v13.9 — RISK ENGINE

Purpose:
- Combine elbow legality + action quality into a single risk assessment
- Deterministic, conservative, ICC-aligned
- Never invent risk when data is missing

Inputs:
- ctx.biomech.elbow (extension + confidence)
- ctx.decision.action_matrix (quality)

Outputs:
- ctx.risk
"""

from app.models.context import Context


# ---------------------------------------------------------
# Risk Weights
# ---------------------------------------------------------
ELBOW_RISK_THRESHOLDS = {
    "SAFE": 0.0,          # ≤ ICC limit
    "MARGINAL": 5.0,      # within error margin
    "ILLEGAL": 10.0,      # clearly illegal
}

ACTION_RISK_MAP = {
    "OPTIMAL": 0.0,
    "GOOD": 0.1,
    "OK": 0.2,
    "SUBOPTIMAL": 0.3,
    "RISK": 0.6,
    "HIGH_RISK": 0.9,
    "UNKNOWN": 0.5,
}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(ctx: Context) -> None:
    """
    Populates ctx.risk with a consolidated injury & legality risk view.
    """

    # -------------------------------------------------
    # Preconditions
    # -------------------------------------------------
    if not ctx.biomech or not ctx.decision:
        ctx.risk = _unknown_risk("Biomechanics or decision state missing")
        return

    elbow = ctx.biomech.elbow
    decision = ctx.decision.action_matrix

    if not elbow or not decision:
        ctx.risk = _unknown_risk("Insufficient data for risk evaluation")
        return

    # -------------------------------------------------
    # Elbow risk evaluation
    # -------------------------------------------------
    ext = elbow.extension_deg
    raw_ext = elbow.extension_raw_deg
    conf = ctx.biomech.elbow_conf or 0.0

    if conf < 80.0:
        elbow_risk = "UNKNOWN"
        elbow_score = 0.5
    elif raw_ext <= ELBOW_RISK_THRESHOLDS["SAFE"]:
        elbow_risk = "SAFE"
        elbow_score = 0.0
    elif raw_ext <= ELBOW_RISK_THRESHOLDS["MARGINAL"]:
        elbow_risk = "MARGINAL"
        elbow_score = 0.4
    else:
        elbow_risk = "ILLEGAL"
        elbow_score = 1.0

    # -------------------------------------------------
    # Action risk evaluation
    # -------------------------------------------------
    action_quality = decision.get("quality", "UNKNOWN")
    action_score = ACTION_RISK_MAP.get(action_quality, 0.5)

    # -------------------------------------------------
    # Combine risk (weighted, conservative)
    # -------------------------------------------------
    # Elbow risk dominates legality
    combined_score = max(elbow_score, action_score)

    if combined_score < 0.25:
        level = "LOW_RISK"
    elif combined_score < 0.6:
        level = "MODERATE_RISK"
    else:
        level = "HIGH_RISK"

    # -------------------------------------------------
    # Store result
    # -------------------------------------------------
    ctx.risk = {
        "score": round(float(combined_score), 3),
        "level": level,
        "details": {
            "elbow_extension_deg": round(float(ext), 2),
            "elbow_raw_extension_deg": round(float(raw_ext), 2),
            "elbow_confidence": round(float(conf), 2),
            "elbow_risk": elbow_risk,
            "action_quality": action_quality,
        },
    }


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _unknown_risk(reason: str):
    return {
        "score": 0.0,
        "level": "UNKNOWN",
        "details": {
            "reason": reason
        },
    }

