from typing import Any, Dict
from app.models.context import Context


def _get_summary(report: Any) -> Dict:
    if report is None:
        return {}
    if isinstance(report, dict):
        return report.get("summary", {})
    return getattr(report, "summary", {}) or {}


def run(ctx: Context) -> Context:
    """
    CUES ENGINE

    Generates actionable coaching cues based on report summary.
    """
    summary = _get_summary(ctx.report)
    legality = summary.get("legality")

    cues = []

    # --------------------------------------
    # Elbow extension coaching cue
    # --------------------------------------
    if legality == "BORDERLINE":
        cues.append(
            "Elbow extension is close or slightly more than the acceptable limit. "
            "Work with a qualified coach to rectify elbow extension "
            "and ensure long-term legality."
        )

    ctx.cues.list = cues
    return ctx

