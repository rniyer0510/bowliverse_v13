# app/risk/lateral_trunk_lean.py

import math
import numpy as np

from app.utils.logger import log
from app.utils.angles import gaussian_smooth


class LateralTrunkLeanRisk:
    """
    Detect abrupt lateral trunk jerk between FFC and UAH.

    Trunk vector = shoulder → pelvis center
    Risk signal   = angular jerk of trunk tilt (unwrapped + smoothed)
    """

    RISK_ID = "LATERAL_TRUNK_LEAN"

    # Thresholds (to be tuned after calibration)
    SPIN_LOW = 0.002
    SPIN_HIGH = 0.004

    PACE_LOW = 0.006
    PACE_HIGH = 0.010

    def compute(self, pose_frames, mapper, ffc, uah, I):
        if uah - ffc < 3:
            return None

        theta = []

        for f in range(ffc, uah + 1):
            pf = pose_frames[f]
            if pf.landmarks is None:
                return None

            try:
                shoulder = mapper.vec(pf.landmarks, "shoulder")
                hip_center = mapper.hip_center(pf.landmarks)
            except Exception as e:
                log(f"[LTL] Landmark error at frame {f}: {e}")
                return None

            vx = shoulder[0] - hip_center[0]
            vy = shoulder[1] - hip_center[1]

            theta.append(math.atan2(vx, vy))

        if len(theta) < 4:
            return None

        # --------------------------------------------------
        # CRITICAL FIXES:
        # 1) unwrap angular discontinuities
        # 2) smooth BEFORE differentiation
        # --------------------------------------------------
        theta = np.unwrap(np.array(theta, dtype=np.float32))
        theta = gaussian_smooth(theta.tolist(), sigma=1.0)

        omega = np.diff(theta)
        jerk = np.diff(omega)

        if len(jerk) == 0:
            return None

        peak_jerk = float(np.max(np.abs(jerk)))

        # Debug (keep until calibration frozen)
        log(f"[DEBUG][LTL] peak_jerk={peak_jerk:.6f} | I={I:.3f}")

        T_low = self._lerp(self.SPIN_LOW, self.PACE_LOW, I)
        T_high = self._lerp(self.SPIN_HIGH, self.PACE_HIGH, I)

        if peak_jerk < T_low:
            level = "LOW"
        elif peak_jerk < T_high:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {
            "risk_id": self.RISK_ID,
            "level": level,
            "metrics": {
                "peak_jerk": round(peak_jerk, 6),
                "window": "FFC→UAH",
                "frames": [ffc, uah],
            },
        }

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

