# app/pipeline/action_matrix.py
"""
ActionLab / Bowliverse v13.9 — ACTION MATRIX

Purpose:
- Convert biomechanical primitives into a single, explainable ACTION STATE
- Deterministic rule-based mapping (no ML, no magic)
- Feeds cues, risk engine, and reports

Inputs (all OPTIONAL but preferred):
- ctx.biomech.hip.zone
- ctx.biomech.shoulder_hip.zone
- ctx.biomech.backfoot.zone

Design rules:
- Works only on zones (not raw angles)
- Graceful degradation if one or more inputs are missing
- Stable across camera setups
"""

from app.models.context import Context


# ---------------------------------------------------------
# Action Matrix Definition
# ---------------------------------------------------------
# Key: (HIP, SHOULDER_HIP, BACKFOOT)
# Value: (ACTION_LABEL, QUALITY_TAG)
#
# QUALITY_TAG is used later by cues / risk engine

ACTION_MATRIX = {
    # ---------------------------------------------
    # IDEAL MECHANICS
    # ---------------------------------------------
    ("SIDE_ON", "HIGH", "CLOSED"): ("ELASTIC_LOAD", "OPTIMAL"),
    ("SIDE_ON", "MODERATE", "CLOSED"): ("CONTROLLED_LOAD", "GOOD"),

    # ---------------------------------------------
    # FRONT-ON STRESS PATTERNS
    # ---------------------------------------------
    ("FRONT_ON", "LOW", "OPEN"): ("FRONT_ON_COLLAPSE", "HIGH_RISK"),
    ("FRONT_ON", "MODERATE", "OPEN"): ("FRONT_ON_FORCE", "RISK"),

    # ---------------------------------------------
    # MIXED ACTION (CLASSIC RED FLAGS)
    # ---------------------------------------------
    ("SIDE_ON", "LOW", "OPEN"): ("MIXED_ACTION", "HIGH_RISK"),
    ("TRANSITIONAL", "LOW", "OPEN"): ("MIXED_ACTION", "HIGH_RISK"),
    ("TRANSITIONAL", "MODERATE", "OPEN"): ("MIXED_ACTION", "RISK"),

    # ---------------------------------------------
    # CONTROLLED BUT SUBOPTIMAL
    # ---------------------------------------------
    ("TRANSITIONAL", "MODERATE", "NEUTRAL"): ("CONTROLLED_ACTION", "OK"),
    ("SIDE_ON", "LOW", "NEUTRAL"): ("LOW_SEPARATION", "SUBOPTIMAL"),

    # ---------------------------------------------
    # FALLBACK SAFE STATES
    # ---------------------------------------------
    ("SIDE_ON", "LOW", "CLOSED"): ("SAFE_BUT_WEAK", "OK"),
    ("FRONT_ON", "HIGH", "NEUTRAL"): ("FORCED_ROTATION", "RISK"),
}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(ctx: Context) -> None:
    """
    Stores result in ctx.decision.action_matrix

    Output structure:
    {
        "action": str,
        "quality": str,
        "inputs": {
            "hip": str | None,
            "shoulder_hip": str | None,
            "backfoot": str | None
        }
    }
    """

    hip_zone = None
    sh_zone = None
    bf_zone = None

    if ctx.biomech.hip:
        hip_zone = ctx.biomech.hip.get("zone")

    if ctx.biomech.shoulder_hip:
        sh_zone = ctx.biomech.shoulder_hip.get("zone")

    if ctx.biomech.backfoot:
        bf_zone = ctx.biomech.backfoot.get("zone")

    # -------------------------------------------------
    # Graceful degradation
    # -------------------------------------------------
    if not hip_zone or not sh_zone or not bf_zone:
        ctx.decision.action_matrix = {
            "action": "INSUFFICIENT_DATA",
            "quality": "UNKNOWN",
            "inputs": {
                "hip": hip_zone,
                "shoulder_hip": sh_zone,
                "backfoot": bf_zone,
            },
        }
        return

    key = (hip_zone, sh_zone, bf_zone)

    if key in ACTION_MATRIX:
        action, quality = ACTION_MATRIX[key]
    else:
        action, quality = ("UNCLASSIFIED_ACTION", "UNKNOWN")

    # -------------------------------------------------
    # Store result
    # -------------------------------------------------
    ctx.decision.action_matrix = {
        "action": action,
        "quality": quality,
        "inputs": {
            "hip": hip_zone,
            "shoulder_hip": sh_zone,
            "backfoot": bf_zone,
        },
    }

