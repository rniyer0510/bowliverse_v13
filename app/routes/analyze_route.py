from fastapi import APIRouter, UploadFile, File
import uuid
from app.models.context import Context

from app.pipeline.input_stage import run as input_stage
from app.pipeline.video_stage import run as video_stage
from app.pipeline.pose_stage import run as pose_stage
from app.pipeline.events_stage import run as events_stage
from app.pipeline.biomech_stage import run as biomech_stage
from app.pipeline.risk_stage import run as risk_stage
from app.pipeline.cues_stage import run as cues_stage
from app.pipeline.report_stage import run as report_stage

router = APIRouter()

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    hand: str = "R",
    bowler_type: str = "pace",
):
    # Save upload to a temp mp4
    suffix = ".mp4" if file.filename.lower().endswith(".mp4") else ""
    tmp_path = f"/tmp/bowliverse_{uuid.uuid4()}{suffix}"

    with open(tmp_path, "wb") as out:
        out.write(await file.read())

    # Build initial context
    ctx = Context(
        input=dict(
            file_path=tmp_path,
            hand=hand.upper(),
            bowler_type=bowler_type.lower(),
        )
    )

    # Pipeline Execution
    ctx = input_stage(ctx)
    ctx = video_stage(ctx)
    ctx = pose_stage(ctx)
    ctx = events_stage(ctx)
    ctx = biomech_stage(ctx)
    ctx = risk_stage(ctx)
    ctx = cues_stage(ctx)
    ctx = report_stage(ctx)

    # Return JSON (no heavy data)
    return ctx.model_dump(
        exclude={
            "video": True,
            "pose": {"frames": True},
        }
    )
