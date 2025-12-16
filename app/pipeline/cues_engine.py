# app/pipeline/cues_engine.py
"""
ActionLab / Bowliverse v13.9 — CUES ENGINE

Purpose:
- Convert ACTION MATRIX output into human-readable coaching cues
- Deterministic, explainable, and safe
- No biomechanical recomputation here — interpretation only

Inputs:
- ctx.decision.action_matrix

Outputs:
- ctx.cues.list : list[str]

Design rules:
- Never invent cues without sufficient data
- Use neutral language for UNKNOWN states
- Avoid technical jargon in final cues
"""

from app.models.context import Context


# ---------------------------------------------------------
# Cue Library
# ---------------------------------------------------------
CUE_LIBRARY = {
    # ---------------------------------------------
    # OPTIMAL / GOOD
    # ---------------------------------------------
    "ELASTIC_LOAD": [
        "Excellent energy transfer through your hips and shoulders.",
        "Your body is well-coiled before release — keep this pattern.",
    ],
    "CONTROLLED_LOAD": [
        "Good control through your lower body.",
        "You are transferring force smoothly into the delivery stride.",
    ],

    # ---------------------------------------------
    # SUBOPTIMAL BUT SAFE
    # ---------------------------------------------
    "SAFE_BUT_WEAK": [
        "Your action is safe but lacks power generation.",
        "Focus on improving separation between hips and shoulders.",
    ],
    "LOW_SEPARATION": [
        "Your shoulders and hips are rotating together.",
        "Try delaying shoulder rotation slightly to store more energy.",
    ],
    "CONTROLLED_ACTION": [
        "Your action is controlled but could be more dynamic.",
        "Small improvements in timing can unlock extra pace.",
    ],

    # ---------------------------------------------
    # RISK / HIGH RISK
    # ---------------------------------------------
    "FRONT_ON_FORCE": [
        "Your body is opening too early during delivery.",
        "This pattern can increase stress on your bowling arm.",
    ],
    "FRONT_ON_COLLAPSE": [
        "Your action shows signs of front-on collapse.",
        "This can overload the shoulder and elbow over time.",
    ],
    "MIXED_ACTION": [
        "Your lower body and upper body are not aligned.",
        "This mixed action increases injury risk — correction is recommended.",
    ],

    # ---------------------------------------------
    # FALLBACKS
    # ---------------------------------------------
    "UNCLASSIFIED_ACTION": [
        "Your action does not match standard patterns.",
        "Further review is needed to give specific guidance.",
    ],
    "INSUFFICIENT_DATA": [
        "Not enough data to generate reliable coaching cues.",
        "Please ensure clear side-on video capture.",
    ],
}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(ctx: Context) -> None:
    """
    Populates ctx.cues.list with coaching cues based on action matrix.
    """

    cues = []

    decision = getattr(ctx, "decision", None)
    if not decision or not getattr(decision, "action_matrix", None):
        ctx.cues.list = [
            "Biomechanical analysis could not be completed.",
            "Please try recording the video again from a clear side-on angle.",
        ]
        return

    action_state = decision.action_matrix
    action = action_state.get("action")

    if not action:
        ctx.cues.list = [
            "Unable to interpret bowling action.",
            "Please consult a coach for a detailed review.",
        ]
        return

    # -------------------------------------------------
    # Lookup cues
    # -------------------------------------------------
    if action in CUE_LIBRARY:
        cues.extend(CUE_LIBRARY[action])
    else:
        cues.extend(CUE_LIBRARY["UNCLASSIFIED_ACTION"])

    # -------------------------------------------------
    # Contextual reinforcement (optional, lightweight)
    # -------------------------------------------------
    quality = action_state.get("quality")
    if quality == "HIGH_RISK":
        cues.append(
            "Consider reducing workload until this action pattern is corrected."
        )
    elif quality == "OPTIMAL":
        cues.append(
            "Maintain this action consistently to reduce injury risk."
        )

    ctx.cues.list = cues

