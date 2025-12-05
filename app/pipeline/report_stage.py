# app/pipeline/report_stage.py
"""
Bowliverse v13.7 — REPORT STAGE

Generates:
    ctx.report.schema_id
    ctx.report.version
    ctx.report.warnings[]
"""

from app.models.context import Context


def run(ctx: Context) -> Context:
    warnings = []

    elbow = ctx.biomech.elbow
    ev = ctx.events

    # ---------------------------------------------------------
    # Elbow warnings
    # ---------------------------------------------------------
    if elbow is None:
        warnings.append("Elbow biomechanics unavailable—check camera angle.")
    else:
        raw = elbow.extension_raw_deg
        final = elbow.extension_deg
        conf = ctx.biomech.elbow_conf

        # If raw is extremely high but confidence is low → angle issue
        if raw > 40 and conf < 85:
            warnings.append("High extension detected but low confidence—video angle may be distorting biomechanics.")

        # If even CWE result > 40 (very rare)
        if final > 40:
            warnings.append("Elbow extension appears high; verify with side-on footage for accuracy.")

    # ---------------------------------------------------------
    # Event ordering warnings
    # ---------------------------------------------------------
    if ev.bfc and ev.ffc and ev.uah and ev.release:
        if not (ev.bfc.frame < ev.ffc.frame < ev.uah.frame < ev.release.frame):
            warnings.append("Event order inconsistent; ensure video is recorded from a proper side-on angle.")

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------
    ctx.report.schema_id = "bowliverse.v13.7"
    ctx.report.version = "13.7.0"
    ctx.report.warnings = warnings

    return ctx

