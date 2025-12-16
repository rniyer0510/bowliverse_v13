# app/pipeline/report_stage.py

from typing import Any, Dict, List
from app.models.context import Context

REPORT_SCHEMA_VERSION = "1.0"


def as_dict(obj: Any) -> Dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


# ---------------------------------------------------------
# Elbow legality classification (ActionLab rule)
# ---------------------------------------------------------
def classify_legality(extension_deg: float | None) -> str:
    """
    ActionLab v1 elbow legality rule:
      < 18°        → LEGAL
      18°–22°      → BORDERLINE
      > 22°        → ILLEGAL
    """
    if extension_deg is None:
        return "INCONCLUSIVE"

    try:
        ext = float(extension_deg)
    except Exception:
        return "INCONCLUSIVE"

    if ext < 18.0:
        return "LEGAL"
    if ext <= 22.0:
        return "BORDERLINE"
    return "ILLEGAL"


def run(ctx: Context) -> Context:
    report: Dict[str, Any] = {"schema_version": REPORT_SCHEMA_VERSION}

    biomech = as_dict(ctx.biomech)
    decision = as_dict(ctx.decision)
    action_matrix = as_dict(decision.get("action_matrix"))
    risk = as_dict(ctx.risk)

    elbow = as_dict(biomech.get("elbow"))
    extension = elbow.get("extension_deg")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    legality = classify_legality(extension)

    report["summary"] = {
        "legality": legality,
        "risk": risk.get("overall", "UNKNOWN"),
        "action_type": action_matrix.get("action", "UNKNOWN"),
        "confidence_pct": float(biomech.get("elbow_conf") or 0.0),
    }

    # -------------------------------------------------
    # Posture table
    # -------------------------------------------------
    posture_table: List[Dict[str, str]] = []

    backfoot = as_dict(biomech.get("backfoot"))
    if backfoot:
        ang = backfoot.get("landing_angle_deg", backfoot.get("angle_deg", 0))
        posture_table.append({
            "key": "Back-foot Angle",
            "value": f"{backfoot.get('zone', '—')} ({round(float(ang or 0), 1)}°)",
            "note": "Orientation at back-foot contact",
        })

    hip = as_dict(biomech.get("hip"))
    if hip:
        posture_table.append({
            "key": "Hip Alignment",
            "value": str(hip.get("zone", "—")),
            "note": f"Hip angle ≈ {round(float(hip.get('angle_deg', 0) or 0), 1)}°",
        })

    shoulder = as_dict(biomech.get("shoulder"))
    if shoulder:
        posture_table.append({
            "key": "Shoulder Alignment",
            "value": str(shoulder.get("zone", "—")),
            "note": f"Shoulder angle ≈ {round(float(shoulder.get('angle_deg', 0) or 0), 1)}°",
        })

    shoulder_hip = as_dict(biomech.get("shoulder_hip"))
    if shoulder_hip:
        sep = shoulder_hip.get("separation_deg", shoulder_hip.get("angle_deg", 0))
        posture_table.append({
            "key": "Hip–Shoulder Separation",
            "value": str(shoulder_hip.get("zone", "—")),
            "note": f"Separation ≈ {round(float(sep or 0), 1)}°",
        })

    report["posture_table"] = posture_table

    # -------------------------------------------------
    # Interpretation
    # -------------------------------------------------
    notes = [
        "Elbow legality is classified using ActionLab thresholds: <18° (Legal), 18–22° (Borderline), >22° (Illegal).",
        "Posture indicators highlight potential risk patterns, not injury diagnosis.",
        "Confidence depends on camera angle and landmark visibility.",
    ]

    if legality == "BORDERLINE":
        notes.append(
            "Borderline elbow extension suggests the action is close to the acceptable limit and should be reviewed with a coach."
        )

    if legality == "ILLEGAL":
        notes.append(
            "Elbow extension exceeds acceptable limits and requires corrective intervention."
        )

    report["interpretation"] = {
        "summary_text": (
            "This report summarizes bowling action legality, posture alignment, "
            "and potential stress indicators using biomechanical checkpoints."
        ),
        "notes": notes,
    }

    # -------------------------------------------------
    # Explainability
    # -------------------------------------------------
    report["explainability"] = {
        "confidence_source": "elbow_conf",
        "extension_deg": extension,
        "legality_rule": "<18° LEGAL | 18–22° BORDERLINE | >22° ILLEGAL",
        "posture_inputs_used": ["backfoot", "hip", "shoulder", "shoulder_hip"],
        "action_matrix_quality": action_matrix.get("quality"),
    }

    ctx.report = report
    return ctx

