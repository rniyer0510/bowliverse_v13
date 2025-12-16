# app/pipeline/decision_state.py
"""
ActionLab / Bowliverse v13.9 — DECISION STATE

Purpose:
- Consolidate ALL downstream interpretation into a single snapshot
- This is the object the frontend should consume
- No computation here — aggregation only

Inputs:
- ctx.events
- ctx.biomech
- ctx.decision.action_matrix
- ctx.risk
- ctx.cues

Outputs:
- ctx.decision.state
"""

from app.models.context import Context


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(ctx: Context) -> None:
    """
    Builds a unified decision state for frontend & reports.
    """

    ctx.decision.state = {
        "events": _safe_events(ctx),
        "biomechanics": _safe_biomech(ctx),
        "action": _safe_action(ctx),
        "risk": ctx.risk if hasattr(ctx, "risk") else None,
        "cues": ctx.cues.list if hasattr(ctx, "cues") else [],
    }


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _safe_events(ctx: Context):
    ev = ctx.events
    if not ev:
        return None

    return {
        "release": _frame(ev.release),
        "uah": _frame(ev.uah),
        "ffc": _frame(ev.ffc),
    }


def _safe_biomech(ctx: Context):
    b = ctx.biomech
    if not b:
        return None

    return {
        "hip": b.hip,
        "shoulder_hip": b.shoulder_hip,
        "backfoot": b.backfoot,
        "elbow": b.elbow,
        "release_height": b.release_height,
    }


def _safe_action(ctx: Context):
    if not ctx.decision or not ctx.decision.action_matrix:
        return None
    return ctx.decision.action_matrix


def _frame(ev):
    if not ev:
        return None
    return {
        "frame": int(ev.frame),
        "confidence": round(float(ev.conf), 2),
    }

