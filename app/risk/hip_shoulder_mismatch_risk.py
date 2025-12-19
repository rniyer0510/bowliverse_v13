import numpy as np
from typing import Dict, Any


class HipShoulderMismatchRisk:
    """
    Detects torsional stress caused by abrupt hip–shoulder separation changes.
    Works for both pacers and spinners using a unified codepath.
    """

    def __init__(self):
        # Conservative initial thresholds (tunable later)
        self.T_SPIN_SEP_CHANGE = 18.0     # degrees
        self.T_PACE_SEP_CHANGE = 12.0     # degrees

        self.T_SPIN_JERK = 250.0          # deg/s^3
        self.T_PACE_JERK = 180.0          # deg/s^3

    @staticmethod
    def interpolate(t_spin: float, t_pace: float, I: float) -> float:
        return (1.0 - I) * t_spin + I * t_pace

    @staticmethod
    def derivative(x: np.ndarray, dt: float) -> np.ndarray:
        return np.gradient(x, dt)

    def compute(
        self,
        hip_angles: np.ndarray,
        shoulder_angles: np.ndarray,
        events: Dict[str, int],
        dt: float,
        I: float
    ) -> Dict[str, Any]:

        # -----------------------------
        # Safety checks
        # -----------------------------
        if "BFC" not in events or "FFC" not in events:
            return self._empty("Missing BFC/FFC events")

        bfc = events["BFC"]
        ffc = events["FFC"]

        if ffc <= bfc + 2:
            return self._empty("Insufficient frames between BFC and FFC")

        # -----------------------------
        # Core signals
        # -----------------------------
        hip = hip_angles[bfc:ffc]
        shoulder = shoulder_angles[bfc:ffc]

        separation = shoulder - hip  # degrees

        # Derivatives
        sep_vel = self.derivative(separation, dt)
        sep_acc = self.derivative(sep_vel, dt)
        sep_jerk = self.derivative(sep_acc, dt)

        # -----------------------------
        # Metrics
        # -----------------------------
        sep_change = float(np.max(separation) - np.min(separation))
        peak_jerk = float(np.max(np.abs(sep_jerk)))

        # -----------------------------
        # Adaptive thresholds
        # -----------------------------
        T_sep = self.interpolate(
            self.T_SPIN_SEP_CHANGE,
            self.T_PACE_SEP_CHANGE,
            I
        )

        T_jerk = self.interpolate(
            self.T_SPIN_JERK,
            self.T_PACE_JERK,
            I
        )

        # -----------------------------
        # Decision logic
        # -----------------------------
        triggered = (
            sep_change > T_sep and
            peak_jerk > T_jerk
        )

        level = "HIGH" if triggered else "LOW"

        return {
            "risk_id": "HIP_SHOULDER_MISMATCH",
            "level": level,
            "triggered": triggered,
            "evidence": {
                "separation_change_deg": round(sep_change, 2),
                "peak_jerk_deg_s3": round(peak_jerk, 2),
                "thresholds": {
                    "sep_change": round(T_sep, 2),
                    "jerk": round(T_jerk, 2)
                },
                "style_intensity_I": round(I, 2)
            },
            "notes": self._explain(triggered, I)
        }

    def _explain(self, triggered: bool, I: float) -> str:
        if not triggered:
            return (
                "Hip and shoulder rotation are well coordinated. "
                "No abrupt torsional load detected."
            )

        if I < 0.35:
            return (
                "Pelvis rotation accelerates abruptly during the front-foot pivot. "
                "This pattern is commonly linked to groin strain in spin bowlers."
            )

        return (
            "Hip–shoulder separation changes abruptly before front-foot bracing. "
            "This mixed-action pattern can stress the lower back and groin."
        )

    @staticmethod
    def _empty(reason: str) -> Dict[str, Any]:
        return {
            "risk_id": "HIP_SHOULDER_MISMATCH",
            "level": "UNKNOWN",
            "triggered": False,
            "evidence": {},
            "notes": reason
        }
