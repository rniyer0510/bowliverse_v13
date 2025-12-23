# app/pipeline/action_matrix.py
"""
Target-2: Action Classification (Priority-1)

Principles:
- Hip = primary anchor
- Back-foot = secondary anchor (can be soft-weighted)
- Shoulder = descriptive only (energy transfer)
- Never force classification on weak evidence
"""

from typing import Dict


# -----------------------------
# Tunables (design-level)
# -----------------------------
MIN_BACKFOOT_SUPPORT = 8
MAX_BACKFOOT_UNCERTAINTY = 35.0

MIN_HIP_SUPPORT = 6


def run(ctx) -> None:
    b = ctx.biomech
    out = {
        "action": "INSUFFICIENT_DATA",
        "confidence_pct": 0,
        "basis": {},
    }

    if not b:
        ctx.decision.action_matrix = out
        return

    hip = getattr(b, "hip", None)
    backfoot = getattr(b, "backfoot", None)

    # -----------------------------
    # HIP evaluation (primary)
    # -----------------------------
    hip_state = "AMBIGUOUS"
    hip_ok = False

    if hip:
        support = hip.get("support_frames", 0)
        angle = hip.get("angle_deg")

        if support >= MIN_HIP_SUPPORT and angle is not None:
            hip_ok = True
            if abs(angle) < 35:
                hip_state = "OPEN"
            else:
                hip_state = "CLOSED"

    out["basis"]["hip"] = hip_state

    # -----------------------------
    # BACK-FOOT evaluation (soft)
    # -----------------------------
    bf_state = "AMBIGUOUS"
    bf_soft_ok = False

    if backfoot:
        bf_support = backfoot.get("support_frames", 0)
        bf_unc = backfoot.get("uncertainty_deg", 999)
        bf_angle = backfoot.get("angle_deg")

        if (
            bf_support >= MIN_BACKFOOT_SUPPORT
            and bf_unc <= MAX_BACKFOOT_UNCERTAINTY
            and bf_angle is not None
        ):
            bf_soft_ok = True
            # toe facing batsman ≈ closed
            if bf_angle > 90:
                bf_state = "CLOSED"
            else:
                bf_state = "OPEN"

    out["basis"]["backfoot"] = bf_state

    # -----------------------------
    # Action resolution
    # -----------------------------
    if hip_ok:
        if hip_state == "OPEN" and bf_soft_ok:
            if bf_state == "OPEN":
                out["action"] = "FRONT_ON"
                out["confidence_pct"] = 55
            else:
                out["action"] = "MIXED"
                out["confidence_pct"] = 45

        elif hip_state == "CLOSED" and bf_soft_ok:
            if bf_state == "CLOSED":
                out["action"] = "SIDE_ON"
                out["confidence_pct"] = 55
            else:
                out["action"] = "MIXED"
                out["confidence_pct"] = 45

        else:
            # Hip-only fallback (explicitly low confidence)
            out["action"] = (
                "FRONT_ON" if hip_state == "OPEN" else "SIDE_ON"
            )
            out["confidence_pct"] = 35

    ctx.decision.action_matrix = out

