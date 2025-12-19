# app/risk/formatter.py

from typing import Dict, Any


def _risk_envelope(
    *,
    core: Dict[str, Any],
    confidence_pct: int = 0,
    window: Dict[str, Any] | None = None,
    evidence: Dict[str, Any] | None = None,
    guardrails_applied: list | None = None,
) -> Dict[str, Any]:
    """
    Additive wrapper that enriches an existing formatted risk
    with stable, machine-readable fields.

    This does NOT change narrative or UI-facing fields.
    """
    return {
        **core,
        "confidence_pct": int(confidence_pct),
        "window": window or {},
        "evidence": evidence or {},
        "guardrails_applied": guardrails_applied or [],
    }


def format_risk(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw risk dict from risk calculators into the standardized
    reporting format used by the frontend and report generator.

    This function is intentionally narrative-first and coach-friendly.
    Machine-readable evidence is added additively via _risk_envelope.
    """

    rid = result["risk_id"]
    level = result.get("level")

    # --------------------------------------------------
    # Explicit SKIPPED handling (no silent drops)
    # --------------------------------------------------
    if level == "SKIPPED":
        return {
            "id": rid,
            "level": "SKIPPED",
            "status": "SKIPPED",
            "summary": "Not enough reliable data to assess this risk.",
            "detail": {
                "metric": "Data sufficiency",
                "value": "Insufficient",
                "why_it_matters": (
                    "Risk warnings are suppressed when evidence is unreliable "
                    "to avoid false positives."
                ),
            },
            "confidence_pct": 0,
            "window": result.get("window", {}),
            "evidence": result.get("evidence", {}),
            "guardrails_applied": result.get("guardrails_applied", []),
        }

    # --------------------------------------------------
    # Hip–Shoulder Mismatch
    # --------------------------------------------------
    if rid == "HIP_SHOULDER_MISMATCH":
        if level == "LOW":
            core = {
                "id": rid,
                "level": "LOW",
                "status": "OK",
                "summary": "Hips and shoulders move in sync.",
                "detail": {
                    "metric": "Segment coordination",
                    "value": "Stable",
                    "why_it_matters": (
                        "Abrupt hip–shoulder mismatch can stress the groin and lower back."
                    ),
                },
            }
        else:
            core = {
                "id": rid,
                "level": level,
                "status": "ATTENTION",
                "summary": "Abrupt hip–shoulder rotation detected.",
                "detail": {
                    "metric": "Segment coordination",
                    "value": "Abrupt change",
                    "why_it_matters": (
                        "Sudden torsional loading during front-foot contact "
                        "can stress the groin and back."
                    ),
                },
            }

        return _risk_envelope(
            core=core,
            confidence_pct=result.get("confidence_pct", 0),
            window=result.get("window"),
            evidence=result.get("evidence"),
            guardrails_applied=result.get("guardrails_applied"),
        )

    # --------------------------------------------------
    # Lateral Trunk Lean
    # --------------------------------------------------
    if rid == "LATERAL_TRUNK_LEAN":
        if level == "LOW":
            core = {
                "id": rid,
                "level": "LOW",
                "status": "OK",
                "summary": "Trunk remains stable through delivery.",
                "detail": {
                    "metric": "Lateral trunk control",
                    "value": "Stable",
                    "why_it_matters": (
                        "A stable trunk allows force to transfer efficiently through the front leg."
                    ),
                },
            }
        else:
            core = {
                "id": rid,
                "level": level,
                "status": "ATTENTION",
                "summary": "Abrupt sideways trunk movement detected after front-foot contact.",
                "detail": {
                    "metric": "Lateral trunk control",
                    "value": "Jerky side movement",
                    "why_it_matters": (
                        "A sudden sideways trunk snap increases load on the lower back "
                        "and reduces control."
                    ),
                },
            }

        return _risk_envelope(
            core=core,
            confidence_pct=result.get("confidence_pct", 0),
            window=result.get("window"),
            evidence=result.get("evidence"),
            guardrails_applied=result.get("guardrails_applied"),
        )

    # --------------------------------------------------
    # Front-Foot Braking Shock
    # --------------------------------------------------
    if rid == "FRONT_FOOT_BRAKING":
        if level == "LOW":
            core = {
                "id": rid,
                "level": "LOW",
                "status": "OK",
                "summary": "Front-foot landing absorbs force smoothly.",
                "detail": {
                    "metric": "Braking control",
                    "value": "Smooth deceleration",
                    "why_it_matters": (
                        "Smooth braking reduces load on the knee, hip, and lower back."
                    ),
                },
            }
        else:
            core = {
                "id": rid,
                "level": level,
                "status": "ATTENTION",
                "summary": "Abrupt front-foot braking detected.",
                "detail": {
                    "metric": "Braking control",
                    "value": "Sudden stop",
                    "why_it_matters": (
                        "A sudden stop at front-foot contact increases joint and spinal load."
                    ),
                },
            }

        return _risk_envelope(
            core=core,
            confidence_pct=result.get("confidence_pct", 0),
            window=result.get("window"),
            evidence=result.get("evidence"),
            guardrails_applied=result.get("guardrails_applied"),
        )

    # --------------------------------------------------
    # Front-Knee Collapse
    # --------------------------------------------------
    if rid == "FRONT_KNEE_COLLAPSE":
        if level == "LOW":
            core = {
                "id": rid,
                "level": "LOW",
                "status": "OK",
                "summary": "Front knee braces and stabilizes after landing.",
                "detail": {
                    "metric": "Front-knee stability",
                    "value": "Stable",
                    "why_it_matters": (
                        "A stable front leg allows force to transfer upward safely."
                    ),
                },
            }
        else:
            core = {
                "id": rid,
                "level": level,
                "status": "ATTENTION",
                "summary": "Front knee continues collapsing after landing.",
                "detail": {
                    "metric": "Front-knee stability",
                    "value": "Delayed brace / collapse",
                    "why_it_matters": (
                        "Failure to brace the front knee increases load on the knee, hip, "
                        "and lower back."
                    ),
                },
            }

        return _risk_envelope(
            core=core,
            confidence_pct=result.get("confidence_pct", 0),
            window=result.get("window"),
            evidence=result.get("evidence"),
            guardrails_applied=result.get("guardrails_applied"),
        )

    # --------------------------------------------------
    # Fallback: Unknown risk ID
    # --------------------------------------------------
    raise ValueError(f"Unknown risk_id: {rid!r}")

