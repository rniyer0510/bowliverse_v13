# app/pipeline/cues_engine.py

from typing import Dict, List
from app.models.context import Context


def run(ctx: Context) -> Context:
    """
    CUES ENGINE (Biomech-driven)

    - Consumes ctx.biomech ONLY (never ctx.report)
    - Produces simple, human coaching cues
    - Designed for kids, grassroots, match & nets
    - No hardcoded defaults
    """

    biomech = ctx.biomech
    cues: List[str] = []

    if not biomech or not biomech.risk:
        ctx.cues.list = []
        return ctx

    risks = biomech.risk.get("breakdown", [])

    backfoot = biomech.backfoot or {}
    hip = biomech.hip or {}
    shoulder = biomech.shoulder or {}
    shoulder_hip = biomech.shoulder_hip or {}

    # -------------------------------------------------
    # PRIORITY 1: Injury / load related cues
    # -------------------------------------------------
    for r in risks:
        level = r.get("level")
        rid = r.get("id")

        if level not in ("MEDIUM", "HIGH"):
            continue

        if rid == "FRONT_FOOT_BRAKING":
            cues.append(
                "Let the front foot take your weight, then complete your follow through"
            )
            break

        if rid == "FRONT_KNEE_COLLAPSE":
            cues.append(
                "Let the front leg support the action without locking it straight"
            )
            break

        if rid == "LATERAL_TRUNK_LEAN":
            cues.append(
                "Keep the body upright as you move through the action"
            )
            break

    # -------------------------------------------------
    # PRIORITY 2: Posture-based cues (no injury yet)
    # -------------------------------------------------
    if not cues:
        bf_zone = backfoot.get("zone")
        hip_zone = hip.get("zone")
        sh_zone = shoulder.get("zone")
        sep_zone = shoulder_hip.get("zone")

        if bf_zone == "VERY_OPEN" and hip_zone == "FRONT_ON" and sep_zone == "LOW":
            cues.append(
                "Allow the body to rotate more freely instead of holding it stiff"
            )

        elif sep_zone == "LOW" and sh_zone == "MIXED":
            cues.append(
                "Let the shoulders move more naturally with the hips"
            )

        elif bf_zone == "VERY_CLOSED":
            cues.append(
                "Give yourself a little more space with the back foot"
            )

    # -------------------------------------------------
    # PRIORITY 3: Smoothness / flow cue (safe fallback)
    # -------------------------------------------------
    if not cues:
        cues.append(
            "Complete your follow through smoothly"
        )

    ctx.cues.list = cues
    return ctx

