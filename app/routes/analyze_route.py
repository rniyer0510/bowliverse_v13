from fastapi import APIRouter, UploadFile, File
import uuid
from app.models.context import Context

from app.pipeline.input_stage import run as input_stage
from app.pipeline.video_stage import run as video_stage
from app.pipeline.pose_stage import run as pose_stage
from app.pipeline.events_stage import run as events_stage
from app.pipeline.biomech_stage import run as biomech_stage
from app.pipeline.action_matrix import run as action_matrix
from app.pipeline.risk_engine import run as risk_engine
from app.pipeline.cues_engine import run as cues_engine
from app.pipeline.report_stage import run as report_stage

router = APIRouter()


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    hand: str = "R",
    bowler_type: str = "pace",
):
    suffix = ".mp4" if file.filename.lower().endswith(".mp4") else ""
    tmp_path = f"/tmp/bowliverse_{uuid.uuid4()}{suffix}"

    with open(tmp_path, "wb") as out:
        out.write(await file.read())

    ctx = Context(
        input=dict(
            file_path=tmp_path,
            hand=hand.upper(),
            bowler_type=bowler_type.lower(),
        )
    )

    ctx = input_stage(ctx)
    ctx = video_stage(ctx)
    ctx = pose_stage(ctx)
    ctx = events_stage(ctx)
    ctx = biomech_stage(ctx)

    # ✅ TARGET-2 ACTION CLASSIFICATION
    action_matrix(ctx)

    ctx = risk_engine(ctx)
    ctx = cues_engine(ctx)
    ctx = report_stage(ctx)

    return ctx.model_dump(
        exclude={
            "video": True,
            "pose": {"frames": True},
        },
        exclude_none=True,
    )

