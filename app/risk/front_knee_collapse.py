import numpy as np
from app.utils.logger import log
from app.risk.signal_quality import signal_quality


class FrontKneeCollapseRisk:
    """
    Front Knee Collapse (FNC)

    Correct biomechanical interpretation:
    - Knee MUST be braced by RELEASE
    - Post-release flexion is allowed (follow-through)
    - Collapse is flagged only if brace is NOT established by release
    """

    RISK_ID = "FRONT_KNEE_COLLAPSE"

    # Angle thresholds (degrees)
    BRACE_ANGLE = 100.0      # knee considered braced if >= this
    MEDIUM_DELTA = 25.0
    HIGH_DELTA = 45.0

    def compute(self, pose_frames, mapper, ffc, release, fps, I):
        # ---------------------------------------------
        # Guardrail: valid window
        # ---------------------------------------------
        if ffc is None or release is None or release <= ffc:
            return self._skipped(["invalid_event_window"])

        # ---------------------------------------------
        # Extract knee angles
        # ---------------------------------------------
        angles = []

        for f in range(ffc, min(len(pose_frames), release + 20)):
            pf = pose_frames[f]
            if pf.landmarks is None:
                continue

            try:
                hip = mapper.vec(pf.landmarks, "hip")
                knee = mapper.vec(pf.landmarks, "knee")
                ankle = mapper.vec(pf.landmarks, "ankle")
            except KeyError:
                continue

            v1 = hip - knee
            v2 = ankle - knee

            ang = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(v1, v2)
                        / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9),
                        -1.0,
                        1.0,
                    )
                )
            )
            angles.append((f, ang))

        if len(angles) < 5:
            return self._skipped(["signal_quality_low"])

        frames, values = zip(*angles)

        if signal_quality(values) < 0.5:
            return self._skipped(["signal_quality_low"])

        # ---------------------------------------------
        # Key angles
        # ---------------------------------------------
        angle_ffc = values[0]

        # closest frame to release
        angle_rel = min(
            values,
            key=lambda a: abs(frames[values.index(a)] - release),
        )

        post_angles = values[values.index(angle_rel):]
        min_post = min(post_angles)

        delta = angle_ffc - min_post

        # ---------------------------------------------
        # CORE FIX: brace must exist BY RELEASE
        # ---------------------------------------------
        if angle_rel >= self.BRACE_ANGLE:
            log(
                f"[DEBUG][FNC] brace_ok "
                f"angle_ffc={angle_ffc:.2f} "
                f"angle_rel={angle_rel:.2f}"
            )
            level = "LOW"
        else:
            if delta >= self.HIGH_DELTA:
                level = "HIGH"
            elif delta >= self.MEDIUM_DELTA:
                level = "MEDIUM"
            else:
                level = "LOW"

            log(
                f"[DEBUG][FNC] angle_ffc={angle_ffc:.2f} "
                f"angle_rel={angle_rel:.2f} "
                f"delta={delta:.2f} "
                f"level={level}"
            )

        return {
            "risk_id": self.RISK_ID,
            "level": level,
            "window": {
                "event": "FFC",
                "frames": [ffc, release + 20],
            },
            "evidence": {
                "angle_at_ffc": round(angle_ffc, 2),
                "angle_at_release": round(angle_rel, 2),
                "min_post_angle": round(min_post, 2),
                "delta": round(delta, 2),
            },
            "guardrails_applied": [
                "release_anchored",
                "brace_checked_by_release",
                "follow_through_allowed",
            ],
        }

    def _skipped(self, reasons):
        return {
            "risk_id": self.RISK_ID,
            "level": "SKIPPED",
            "guardrails_applied": reasons,
        }

