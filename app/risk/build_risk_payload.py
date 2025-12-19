from app.risk.hip_shoulder_mismatch_risk import HipShoulderMismatchRisk
from app.risk.formatter import format_risk
from app.risk.aggregator import aggregate_risks


def build_risk_payload(context):
    hip_risk = HipShoulderMismatchRisk().compute(
        hip_angles=context["hip_angles"],
        shoulder_angles=context["shoulder_angles"],
        events=context["events"],
        dt=context["dt"],
        I=context["style_I"]
    )

    raw_risks = [hip_risk]

    formatted = [format_risk(r) for r in raw_risks]

    return {
        "overall": aggregate_risks(formatted),
        "breakdown": formatted
    }
