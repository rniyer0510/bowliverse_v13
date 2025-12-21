# app/explainability/p3_engine.py

from typing import Dict, Any, List


class P3Engine:
    """
    Refined 3P3 engine using unified intensity scalar I.
    - No pace/spin branching
    - Baseball-style rate-aware biomechanics
    """

    # -------------------------------------------------
    # Public entry
    # -------------------------------------------------
    def compute(
        self,
        biomech: Dict[str, Any],
        risks: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        I = self._compute_intensity(biomech, risks)

        posture = self._compute_posture(biomech, I)
        power = self._compute_power(biomech, risks, I)
        protection = self._compute_protection(risks, I)

        return {
            "intensity": round(I, 2),
            "posture": posture,
            "power": power,
            "protection": protection,
        }

    # -------------------------------------------------
    # Intensity scalar I
    # -------------------------------------------------
    def _compute_intensity(self, biomech: Dict[str, Any], risks: Dict[str, Any]) -> float:
        vals: List[float] = []

        # Front-foot braking impulse
        b = self._get_risk_evidence(risks, "FRONT_FOOT_BRAKING", "braking_index")
        if b is not None:
            vals.append(max(0.0, min(1.0, b)))

        # Hip–shoulder separation proxy
        sep = biomech.get("shoulder_hip", {}).get("separation_deg")
        if sep is not None:
            vals.append(min(abs(float(sep)) / 45.0, 1.0))

        # Deceleration balance proxy
        e = self._get_risk_evidence(risks, "FRONT_FOOT_BRAKING", "early_decel")
        l = self._get_risk_evidence(risks, "FRONT_FOOT_BRAKING", "late_decel")
        if e is not None and l is not None:
            imbalance = abs(e - l)
            vals.append(max(0.0, 1.0 - imbalance))

        if not vals:
            return 0.3  # conservative fallback

        return max(0.0, min(1.0, sum(vals) / len(vals)))

    # -------------------------------------------------
    # POSTURE (intensity-aware)
    # -------------------------------------------------
    def _compute_posture(self, biomech: Dict[str, Any], I: float) -> Dict[str, Any]:
        score = 1.0
        signals: List[str] = []

        penalty_scale = 0.6 + 0.4 * I  # instability hurts more at high I

        backfoot = biomech.get("backfoot", {})
        if backfoot.get("zone") in ("VERY_OPEN", "VERY_CLOSED"):
            score -= 0.15 * penalty_scale
            signals.append("Weak back-foot base")

        shoulder = biomech.get("shoulder", {})
        if shoulder.get("zone") in ("VERY_OPEN", "VERY_CLOSED"):
            score -= 0.20 * penalty_scale
            signals.append("Unstable shoulder alignment")

        score = max(0.0, min(1.0, score))
        return self._label(score, signals)

    # -------------------------------------------------
    # POWER (intensity-relative)
    # -------------------------------------------------
    def _compute_power(self, biomech: Dict[str, Any], risks: Dict[str, Any], I: float) -> Dict[str, Any]:
        score = 1.0
        signals: List[str] = []

        braking = self._get_risk_evidence(risks, "FRONT_FOOT_BRAKING", "braking_index")
        if braking is not None:
            expected = 0.35 + 0.35 * I
            if braking < expected:
                score -= 0.30 * I
                signals.append("Energy transfer below intensity expectation")

        sep = biomech.get("shoulder_hip", {}).get("separation_deg")
        if sep is not None and abs(float(sep)) < 5.0 and I > 0.5:
            score -= 0.15
            signals.append("Limited hip–shoulder contribution for intensity")

        score = max(0.0, min(1.0, score))
        return self._label(score, signals)

    # -------------------------------------------------
    # PROTECTION (intensity-amplified)
    # -------------------------------------------------
    def _compute_protection(self, risks: Dict[str, Any], I: float) -> Dict[str, Any]:
        score = 1.0
        signals: List[str] = []

        for item in risks.get("breakdown", []):
            lvl = item.get("level")
            rid = item.get("id")

            if lvl == "HIGH":
                score -= 0.50 * (0.7 + 0.3 * I)
                signals.append(f"High load risk at intensity: {rid}")
            elif lvl == "MODERATE":
                score -= 0.25 * (0.6 + 0.4 * I)
                signals.append(f"Moderate load risk at intensity: {rid}")

        score = max(0.0, min(1.0, score))
        return self._label(score, signals)

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _label(self, score: float, signals: List[str]) -> Dict[str, Any]:
        if score >= 0.80:
            label = "EXCELLENT"
        elif score >= 0.65:
            label = "GOOD"
        elif score >= 0.45:
            label = "NEEDS_WORK"
        else:
            label = "POOR"

        return {
            "score": round(score, 2),
            "label": label,
            "signals": signals,
        }

    def _get_risk_evidence(self, risks: Dict[str, Any], rid: str, key: str):
        for item in risks.get("breakdown", []):
            if item.get("id") == rid:
                return item.get("evidence", {}).get(key)
        return None
