# app/pipeline/report_stage.py

from typing import Any, Dict, List
from app.models.context import Context
from app.explainability.p3_engine import P3Engine

REPORT_SCHEMA_VERSION = "1.0"
REPORT_FULL_SCHEMA = "report_full.v1"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Main Report Builder
# ---------------------------------------------------------
def run(ctx: Context) -> Context:
    report: Dict[str, Any] = {"schema_version": REPORT_SCHEMA_VERSION}

    biomech = as_dict(ctx.biomech)
    decision = as_dict(ctx.decision)
    action_matrix = as_dict(decision.get("action_matrix"))
    risk = as_dict(ctx.risk)

    elbow = as_dict(biomech.get("elbow"))
    extension = elbow.get("extension_deg")

    # -------------------------------------------------
    # Core legality + P3 computation
    # -------------------------------------------------
    legality = classify_legality(extension)

    p3_engine = P3Engine()
    p3 = p3_engine.compute(
        biomech=biomech,
        risks=risk,
        context=as_dict(getattr(ctx, "context", None)),
    )

    # -------------------------------------------------
    # Frontend Summary (minimal, stable)
    # -------------------------------------------------
    report["summary"] = {
        "action_type": action_matrix.get("action", "UNKNOWN"),
        "legality": legality,
        "overall_risk": risk.get("overall", "UNKNOWN"),
        "posture": p3.get("posture", {}).get("label"),
        "power": p3.get("power", {}).get("label"),
        "protection": p3.get("protection", {}).get("label"),
        "confidence_pct": float(biomech.get("elbow_conf") or 0.0),
    }

    # -------------------------------------------------
    # Posture Table (UI-friendly, derived)
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
    # Interpretation (derived, neutral)
    # -------------------------------------------------
    report["interpretation"] = {
        "summary_text": (
            "This report summarizes bowling action characteristics using "
            "Posture, Power, and Protection indicators derived strictly "
            "from biomechanical and risk signals."
        ),
        "notes": [
            "All interpretations are derived from measured biomechanical data.",
            "No coaching advice is generated without supporting evidence.",
        ],
    }

    # -------------------------------------------------
    # Explainability (audit layer)
    # -------------------------------------------------
    report["explainability"] = {
        "confidence_source": "elbow_conf",
        "extension_deg": extension,
        "legality_rule": "<18° LEGAL | 18–22° BORDERLINE | >22° ILLEGAL",
        "posture_inputs_used": ["backfoot", "hip", "shoulder", "shoulder_hip"],
        "action_matrix_quality": action_matrix.get("quality"),
    }

    # -------------------------------------------------
    # 3P3 (compact, frontend-ready)
    # -------------------------------------------------
    report["p3"] = p3

    # -------------------------------------------------
    # Full Report (PDF / audit / long-form)
    # -------------------------------------------------
    risks_breakdown = risk.get("breakdown", [])

    evaluated = []
    suppressed = []

    for r in risks_breakdown:
        if r.get("status") == "SKIPPED":
            suppressed.append({
                "risk": r.get("id"),
                "reason": r.get("summary"),
            })
        else:
            evaluated.append({
                "risk": r.get("id"),
                "level": r.get("level"),
                "summary": r.get("summary"),
            })

    report["report_full"] = {
        "meta": {
            "schema": REPORT_FULL_SCHEMA,
            "model": "ActionLab Phase-1",
            "action_type": action_matrix.get("action"),
            "analysis_confidence_pct": float(biomech.get("elbow_conf") or 0.0),
        },

        "executive_summary": {
            "posture": p3.get("posture", {}).get("label"),
            "power": p3.get("power", {}).get("label"),
            "protection": p3.get("protection", {}).get("label"),
            "overall_risk": risk.get("overall"),
            "limiting_factors": (
                p3.get("posture", {}).get("signals", [])
                + p3.get("power", {}).get("signals", [])
            ),
        },

        "legality": {
            "verdict": legality,
            "uah_angle_deg": elbow.get("uah_angle"),
            "release_angle_deg": elbow.get("release_angle"),
            "computed_extension_deg": elbow.get("extension_deg"),
            "peak_extension_angle_deg": elbow.get("peak_extension_angle_deg"),
            "confidence_pct": float(biomech.get("elbow_conf") or 0.0),
            "error_margin_deg": elbow.get("extension_error_margin_deg"),
        },

        "posture_components": [
            {
                "name": "Back-Foot",
                "angle_deg": backfoot.get("landing_angle_deg"),
                "zone": backfoot.get("zone"),
            },
            {
                "name": "Hip",
                "angle_deg": hip.get("angle_deg"),
                "zone": hip.get("zone"),
            },
            {
                "name": "Shoulder",
                "angle_deg": shoulder.get("angle_deg"),
                "zone": shoulder.get("zone"),
            },
            {
                "name": "Hip–Shoulder Separation",
                "angle_deg": shoulder_hip.get("separation_deg"),
                "zone": shoulder_hip.get("zone"),
            },
        ],

        "power_metrics": {
            "intensity_scalar_I": p3.get("intensity"),
            "front_foot_braking_index": next(
                (r.get("evidence", {}).get("braking_index")
                 for r in risks_breakdown
                 if r.get("id") == "FRONT_FOOT_BRAKING"),
                None,
            ),
        },

        "protection": {
            "evaluated": evaluated,
            "suppressed": suppressed,
        },

        "guidance": {
            "primary_cues": as_dict(ctx.cues).get("list", []),
            "focus_areas": (
                p3.get("posture", {}).get("signals", [])
                + p3.get("power", {}).get("signals", [])
            ),
            "observed_risks": [
                r.get("id")
                for r in risks_breakdown
                if r.get("level") in ("LOW", "MEDIUM", "HIGH")
            ],
        },

        "audit": {
            "events": ["BFC", "FFC", "UAH", "RELEASE"],
            "guardrails": [
                "Unified intensity scalar (pace & spin agnostic)",
                "Signal-quality gating",
                "Derived-only interpretation",
                "Raw risk preserved separately from report interpretation",
            ],
        },
    }

    ctx.report = report
    return ctx

