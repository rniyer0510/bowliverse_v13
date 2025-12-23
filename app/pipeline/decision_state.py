# app/pipeline/decision_state.py
"""
ActionLab / Bowliverse v13.x — DECISION STATE (READ-ONLY)

Purpose:
- Consolidate downstream interpretation into a single snapshot
- MUST NOT compute action classification
- Action classification is produced upstream in pipeline
"""

from app.models.context import Context


def run(ctx: Context) -> None:
    """
    Builds a unified decision snapshot for frontend & reports.
    """

    ctx.decision.state = {
        "events": _safe_events(ctx),
        "biomechanics": _safe_biomech(ctx),
        "action": ctx.decision.action_matrix,
        "risk": ctx.risk,
        "cues": ctx.cues.list,
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
        "bfc": _frame(ev.bfc),
    }


def _safe_biomech(ctx: Context):
    b = ctx.biomech
    if not b:
        return None

    return {
        "hip": b.hip,
        "shoulder": b.shoulder,
        "shoulder_hip": b.shoulder_hip,
        "backfoot": b.backfoot,
        "elbow": b.elbow,
        "release_height": b.release_height,
    }


def _frame(ev):
    if not ev:
        return None
    return {
        "frame": int(ev.frame),
        "confidence": round(float(ev.conf), 2),
    }

