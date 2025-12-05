# app/pipeline/cues_stage.py
"""
Bowliverse v13.5 — Coaching Cues Stage

Purpose:
    Translate biomechanics + risk into human-readable, coaching-oriented
    recommendations.
"""

from app.models.context import Context


def _add(cues, msg):
    if msg not in cues:
        cues.append(msg)


def run(ctx: Context) -> Context:
    cues = []
    biomech = ctx.biomech
    elbow = biomech.elbow
    height = biomech.release_height
    risk = ctx.risk

    # If no biomech → cannot produce cues
    if elbow is None:
        ctx.cues.list = ["Insufficient biomechanical visibility for coaching cues."]
        return ctx

    ext = float(elbow.extension_deg)
    abs_ext = abs(ext)
    ext_conf = biomech.elbow_conf or 0

    # -------------------------  
    # EXTENSION-BASED CUES  
    # -------------------------
    if abs_ext < 10:
        _add(cues, "Your elbow extension is within a safe range.")
        if ext_conf < 70:
            _add(cues, "Visibility was low—consider recording from a clearer angle.")
    elif abs_ext < 20:
        _add(cues, "Monitor your elbow extension; slight technique refinement may help.")
        if ext_conf > 85:
            _add(cues, "Your technique is mostly stable; focus on smoother load-up before release.")
    else:
        _add(cues, "High elbow extension detected; seek corrective coaching.")
        if ext_conf > 85:
            _add(cues, "Strong visibility confirms this is a reliable reading.")
        else:
            _add(cues, "Visibility uncertainty—try recording from the bowling-arm side.")

    # -------------------------
    # RELEASE HEIGHT CUES
    # -------------------------
    if height:
        nh = float(height.norm_height)
        if nh > 0.3:
            _add(cues, "Your release point is quite high; ensure your front-arm pull is controlled.")
        elif nh < -0.2:
            _add(cues, "Low release height detected; consider improving your front-arm stability.")

    # -------------------------
    # RISK-BASED META CUE
    # -------------------------
    if risk.level == "HIGH_RISK":
        _add(cues, "Multiple high-load indicators detected—review your action with a coach.")
    elif risk.level == "MEDIUM_RISK":
        _add(cues, "Moderate risk—focus on repeatability and smooth transition from UAH to Release.")
    elif risk.level == "LOW_RISK":
        _add(cues, "Your action appears biomechanically efficient.")

    ctx.cues.list = cues
    return ctx

