# app/pipeline/action_matrix.py

from app.models.context import Context


def run(ctx: Context) -> None:
    """
    Stores result in ctx.decision.action_matrix

    Lower-half dominant logic:
    - Hip + Backfoot decide ACTION
    - Shoulder–hip separation modifies QUALITY only
    """

    hip = ctx.biomech.hip.get("zone") if ctx.biomech.hip else None
    sh = ctx.biomech.shoulder_hip.get("zone") if ctx.biomech.shoulder_hip else None
    bf = ctx.biomech.backfoot.get("zone") if ctx.biomech.backfoot else None

    # ---------------------------------------------
    # Graceful degradation
    # ---------------------------------------------
    if not hip or not bf:
        ctx.decision.action_matrix = {
            "action": "INSUFFICIENT_DATA",
            "quality": "UNKNOWN",
            "inputs": {"hip": hip, "shoulder_hip": sh, "backfoot": bf},
        }
        return

    # ---------------------------------------------
    # STEP 1: BASE ACTION (LOWER HALF ONLY)
    # ---------------------------------------------
    if hip == "FRONT_ON" and bf in ("OPEN", "VERY_OPEN", "NEUTRAL"):
        action = "FRONT_ON"

    elif hip == "SIDE_ON" and bf == "CLOSED":
        action = "SIDE_ON"

    else:
        action = "MIXED"

    # ---------------------------------------------
    # STEP 2: QUALITY MODULATION (UPPER HALF)
    # ---------------------------------------------
    quality = "OK"

    if sh == "LOW":
        quality = "SUBOPTIMAL"

    elif sh == "HIGH":
        quality = "OPTIMAL"

    # Mixed action is always higher risk
    if action == "MIXED":
        quality = "HIGH_RISK"

    # ---------------------------------------------
    # STORE RESULT
    # ---------------------------------------------
    ctx.decision.action_matrix = {
        "action": action,
        "quality": quality,
        "inputs": {
            "hip": hip,
            "shoulder_hip": sh,
            "backfoot": bf,
        },
    }

