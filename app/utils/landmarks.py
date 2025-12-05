import numpy as np


class LandmarkMapper:
    """
    Bowliverse v13.7 — Landmark Mapper + Baseball Arm-Plane Detector
    """

    def __init__(self, hand: str, vis_threshold: float = 0.20):
        self.hand = hand.upper()
        self.vis_threshold = vis_threshold

        self.left = {
            "shoulder": 11,
            "elbow": 13,
            "wrist": 15,
            "hip": 23,
            "knee": 25,
            "ankle": 27,
            "heel": 29,
            "toe": 31,
        }

        self.right = {
            "shoulder": 12,
            "elbow": 14,
            "wrist": 16,
            "hip": 24,
            "knee": 26,
            "ankle": 28,
            "heel": 30,
            "toe": 32,
        }

        self.primary = self.right if self.hand == "R" else self.left

    # -----------------------------------------------------
    # Safe landmark fetch
    # -----------------------------------------------------

    def _safe_vec(self, lm, idx, fallback=None):
        p = lm[idx]
        vis = float(p.get("vis", 0.0))
        if vis < self.vis_threshold and fallback is not None:
            return fallback
        return np.array([p["x"], p["y"], p["z"]], float)

    # -----------------------------------------------------
    # Arm triplet (W,E,S)
    # -----------------------------------------------------

    def arm_triplet(self, lm, prev=None):
        prev_w = prev["w"] if prev else None
        prev_e = prev["e"] if prev else None
        prev_s = prev["s"] if prev else None

        W = self._safe_vec(lm, self.primary["wrist"], fallback=prev_w)
        E = self._safe_vec(lm, self.primary["elbow"], fallback=prev_e)
        S = self._safe_vec(lm, self.primary["shoulder"], fallback=prev_s)
        return W, E, S

    # -----------------------------------------------------
    # Shoulders & hips
    # -----------------------------------------------------

    def hips_pair(self, lm):
        LH = self._safe_vec(lm, self.left["hip"])
        RH = self._safe_vec(lm, self.right["hip"])
        return LH, RH

    def shoulders_pair(self, lm):
        LS = self._safe_vec(lm, self.left["shoulder"])
        RS = self._safe_vec(lm, self.right["shoulder"])
        return LS, RS

    # -----------------------------------------------------
    # REQUIRED BY EVENTS STAGE (FFC DETECTION)
    # -----------------------------------------------------

    def ankle_y(self, lm):
        """Return ankle Y for the dominant (bowling) leg."""
        idx = self.primary["ankle"]
        return float(lm[idx]["y"])

    def hip_center(self, lm):
        LH, RH = self.hips_pair(lm)
        return (LH + RH) / 2.0

    # -----------------------------------------------------
    # Arm-plane normal (baseball method)
    # -----------------------------------------------------

    def arm_plane_normal(self, lm):
        LS, RS = self.shoulders_pair(lm)
        LH, RH = self.hips_pair(lm)

        shoulder_axis = RS - LS
        hip_axis = RH - LH

        trunk_axis = shoulder_axis + hip_axis
        trunk_axis = trunk_axis / (np.linalg.norm(trunk_axis) + 1e-9)

        global_up = np.array([0.0, 1.0, 0.0], float)

        n = np.cross(trunk_axis, global_up)
        n = n / (np.linalg.norm(n) + 1e-9)

        return n

    # -----------------------------------------------------
    # Generic vector accessor
    # -----------------------------------------------------

    def vec(self, lm, key: str):
        idx = self.primary.get(key)
        if idx is None:
            raise KeyError(f"Invalid vector key: {key}")
        return self._safe_vec(lm, idx)

