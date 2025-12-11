# app/pipeline/elbow_refine_stage.py

from app.models.context import Context
from app.utils.angles import elbow_flexion
from app.utils.landmarks import LandmarkMapper

def run(ctx: Context) -> Context:
    if ctx.biomech.error:
        return ctx

    mapper = LandmarkMapper(ctx.input.hand)

    try:
        ev = ctx.events
        pf = ctx.pose.frames

        f_uah = int(ev.uah.frame)
        f_rel = int(ev.release.frame)

        lm_u = pf[f_uah].landmarks
        lm_r = pf[f_rel].landmarks

        S_u = mapper.vec(lm_u, "shoulder")
        E_u = mapper.vec(lm_u, "elbow")
        W_u = mapper.vec(lm_u, "wrist")

        S_r = mapper.vec(lm_r, "shoulder")
        E_r = mapper.vec(lm_r, "elbow")
        W_r = mapper.vec(lm_r, "wrist")

        # FIX: write correct field names
        ctx.biomech.elbow.uah_angle = float(elbow_flexion(S_u, E_u, W_u))
        ctx.biomech.elbow.release_angle = float(elbow_flexion(S_r, E_r, W_r))

    except Exception:
        pass

    return ctx

