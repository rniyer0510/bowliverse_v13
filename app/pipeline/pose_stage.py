import mediapipe as mp
import numpy as np
from app.models.context import Context
from app.models.pose_model import PoseFrame

mp_pose = mp.solutions.pose


def run(ctx: Context) -> Context:
    """
    Pose stage — Stable v13.7.1:
    - Always preserve frame count.
    - Store PoseFrame even when MediaPipe fails.
    - No crashes when landmarks=None.
    """
    try:
        frames = ctx.video.frames
        if not frames:
            ctx.pose.error = "No video frames provided"
            return ctx

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
        )

        results_list = []

        for idx, frame in enumerate(frames):
            rgb = frame[:, :, ::-1]
            result = pose.process(rgb)

            if not result.pose_landmarks:
                results_list.append(
                    PoseFrame(
                        frame_index=idx,
                        landmarks=None,
                        confidence=0.0
                    )
                )
                continue

            lm = result.pose_landmarks.landmark
            landmarks = [
                {
                    "x": float(p.x),
                    "y": float(p.y),
                    "z": float(p.z),
                    "vis": float(p.visibility),
                }
                for p in lm
            ]

            results_list.append(
                PoseFrame(
                    frame_index=idx,
                    landmarks=landmarks,
                    confidence=float(lm[0].visibility)
                )
            )

        pose.close()
        ctx.pose.frames = results_list
        ctx.pose.total_frames = len(results_list)
        ctx.pose.fps = ctx.video.fps
        ctx.pose.duration_sec = ctx.video.duration_sec
        ctx.pose.error = None

    except Exception as e:
        ctx.pose.error = str(e)

    return ctx
