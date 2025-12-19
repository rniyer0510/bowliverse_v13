"""
ActionLab — Risk Engine (Option A)

Plumbing-only stage.

Responsibilities:
1) Lift ctx.biomech.risk → ctx.risk
2) Aggregate overall risk + confidence
3) Never infer or invent risk
"""

from typing import Any, Dict, List
from app.models.context import Context


def _normalize(v: Any) -> Dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if hasattr(v, "model_dump"):
        return v.model_dump()
    if hasattr(v, "dict"):
        return v.dict()
    return {}


def _aggregate_overall(breakdown: List[Dict[str, Any]]) -> str:
    level = "Low"
    for r in breakdown:
        l = (r.get("level") or "").upper()
        if l == "HIGH":
            return "High"
        if l == "MEDIUM":
            level = "Medium"
    return level


def _confidence(breakdown: List[Dict[str, Any]]) -> float:
    levels = [(r.get("level") or "").upper() for r in breakdown]
    levels = [l for l in levels if l in ("LOW", "MEDIUM", "HIGH")]
    if not levels:
        return 0.0
    dominant = max(levels.count("LOW"), levels.count("MEDIUM"), levels.count("HIGH"))
    return round(dominant / len(levels), 2)


def run(ctx: Context) -> Context:
    biomech_risk = _normalize(getattr(ctx.biomech, "risk", None))
    breakdown = biomech_risk.get("breakdown", []) or []

    ctx.risk.breakdown = breakdown
    ctx.risk.overall = _aggregate_overall(breakdown)
    ctx.risk.confidence = _confidence(breakdown)

    return ctx

