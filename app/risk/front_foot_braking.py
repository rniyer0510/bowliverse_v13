import numpy as np
from app.risk.signal_quality import signal_quality
from app.utils.logger import log


class FrontFootBrakingRisk:
    """
    Practical braking risk:
    - How abruptly speed drops
    - Whether deceleration is spread into follow-through
    """

    RISK_ID = "FRONT_FOOT_BRAKING"

    # Conservative thresholds (empirically safe)
    LOW = 2.5
    HIGH = 5.5

    def compute(self, pose_frames, mapper, ffc, release, fps, I):
        # Window: FFC → Release → Follow-through
        end = min(len(pose_frames) - 1, release + 15)
        xs = []

        for f in range(ffc, end):
            pf = pose_frames[f]
            if pf.landmarks is None:
                # Skip missing frame instead of hard-failing
                continue
            hip = mapper.hip_center(pf.landmarks)
            xs.append(float(hip[0]))

        # Guard against too-short signal (prevents gradient issues)
        if len(xs) < 3:
            return None

        q = signal_quality(xs)
        if q < 0.5:
            return None

        dt = 1.0 / fps
        v = np.gradient(xs, dt)

        # Early vs late decel
        early = max(0.0, v[0] - min(v[: len(v) // 2]))
        late = max(0.0, v[len(v) // 2] - min(v))

        # If follow-through absorbs load, late decel should dominate
        distribution = late / (early + late + 1e-6)
        effective = early * (1.0 - distribution)

        log(f"[DEBUG][FFB] early={early:.3f} late={late:.3f} eff={effective:.3f}")

        if effective < self.LOW:
            level = "LOW"
        elif effective < self.HIGH:
            level = "MEDIUM"
        else:
            level = "HIGH"

        # --------------------------------------------------
        # Additive RiskOutput enrichment (safe)
        # --------------------------------------------------
        pre = int(0.08 * fps)
        post = int(0.12 * fps)

        window_start = max(0, ffc - pre)
        window_end = min(len(pose_frames) - 1, release + post)

        return {
            "risk_id": self.RISK_ID,
            "level": level,

            # Existing metrics (unchanged)
            "metrics": {
                "early_decel": round(early, 3),
                "late_decel": round(late, 3),
                "distribution": round(distribution, 3),
            },

            # Additive, explainable fields
            "confidence_pct": int(q * 100),
            "window": {
                "event": "FFC",
                "frames": [window_start, window_end],
            },
            "evidence": {
                "braking_index": round(effective, 3),
                "early_decel": round(early, 3),
                "late_decel": round(late, 3),
                "distribution": round(distribution, 3),
            },
            "guardrails_applied": [
                "signal_quality_ok",
                "min_window_ok",
                "release_anchored",
            ],
        }

