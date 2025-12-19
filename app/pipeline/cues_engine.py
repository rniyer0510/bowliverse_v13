# app/pipeline/cues_engine.py

from typing import Any, Dict
from app.models.context import Context


def _get_summary(report: Any) -> Dict:
    if report is None:
        return {}
    if isinstance(report, dict):
        return report.get("summary", {})
    return getattr(report, "summary", {}) or {}


def _get_risks(report: Any):
    if report is None:
        return []
    if isinstance(report, dict):
        return report.get("risk", {}).get("breakdown", [])
    return getattr(report, "risk", {}).get("breakdown", [])


def run(ctx: Context) -> Context:
    """
    CUES ENGINE

    Generates ONE simple, shoutable coaching cue.
    Designed for kids, grassroots coaches, and match use.
    """
    summary = _get_summary(ctx.report)
    risks = _get_risks(ctx.report)

    cue = None

    # --------------------------------------
    # Risk-based cues (priority order)
    # --------------------------------------
    for r in risks:
        if r.get("level") in ("LOW", "SKIPPED"):
            continue

        rid = r.get("id")

        if rid == "FRONT_FOOT_BRAKING":
            cue = "Land soft, then flow"
            break

        if rid == "FRONT_KNEE_COLLAPSE":
            cue = "Brace front leg, then flow"
            break

        if rid == "LATERAL_TRUNK_LEAN":
            cue = "Stay tall and balanced"
            break

    # --------------------------------------
    # Elbow cue (only if no body cue chosen)
    # --------------------------------------
    if cue is None:
        legality = summary.get("legality")
        if legality == "BORDERLINE":
            cue = "Relax the arm, stay smooth"

    # --------------------------------------
    # Default cue
    # --------------------------------------
    if cue is None:
        cue = "Stay tall and balanced"

    ctx.cues.list = [cue]
    return ctx

