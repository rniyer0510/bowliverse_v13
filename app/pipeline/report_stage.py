# app/pipeline/report_stage.py
"""
Bowliverse v13.7 — REPORT STAGE (Frontend-lean)

Now we only populate:
    ctx.report.warnings  -> List[str]

We deliberately DO NOT set:
    - ctx.report.schema_id
    - ctx.report.version

Those fields are no longer required for the frontend JSON and are
kept unused to avoid bloating the response.
"""

from app.models.context import Context


def run(ctx: Context) -> Context:
    warnings: list[str] = []

    elbow = ctx.biomech.elbow
    ev = ctx.events

    # ---------------------------------------------------------
    # Elbow-related warnings
    # ---------------------------------------------------------
    if elbow is None:
        warnings.append("Elbow biomechanics unavailable—check camera angle.")
    else:
        raw = getattr(elbow, "extension_raw_deg", None)
        final = getattr(elbow, "extension_deg", None)
        conf = getattr(ctx.biomech, "elbow_conf", None)

        # High raw extension but low confidence -> likely angle issue
        try:
            if raw is not None and conf is not None:
                if raw > 40 and conf < 85:
                    warnings.append(
                        "High extension detected but low confidence—video angle may be distorting biomechanics."
                    )
        except TypeError:
            # Graceful degradation if types are unexpected
            pass

        # If even the cleaned ICC-style extension is very high
        try:
            if final is not None and final > 40:
                warnings.append(
                    "Elbow extension appears high; verify with side-on footage for accuracy."
                )
        except TypeError:
            pass

    # ---------------------------------------------------------
    # Event ordering warnings
    # ---------------------------------------------------------
    try:
        if ev and ev.bfc and ev.ffc and ev.uah and ev.release:
            if not (ev.bfc.frame < ev.ffc.frame < ev.uah.frame < ev.release.frame):
                warnings.append(
                    "Event order inconsistent; ensure video is recorded from a proper side-on angle."
                )
    except Exception:
        # Any unexpected shape in events → just skip ordering warning
        pass

    # ---------------------------------------------------------
    # Save report (minimal)
    # ---------------------------------------------------------
    ctx.report.warnings = warnings
    # NOTE: schema_id and version are intentionally NOT set anymore

    return ctx

