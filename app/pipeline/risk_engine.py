"""
ActionLab — Risk Engine
Posture-driven aggregation (conservative)
"""

from app.models.context import Context


def run(ctx: Context) -> Context:
    risks = []

    try:
        for row in ctx.report.posture_table:
            if row.parameter == "Front knee stability":
                ctx.risk.knee = row.risk
                if row.risk == "Medium":
                    risks.append("Medium")
    except Exception:
        pass

    if "High" in risks:
        ctx.risk.overall = "High"
    elif "Medium" in risks:
        ctx.risk.overall = "Medium"
    else:
        ctx.risk.overall = "Low"

    return ctx
