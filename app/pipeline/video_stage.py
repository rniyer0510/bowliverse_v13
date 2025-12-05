import cv2
from app.models.context import Context

def run(ctx: Context) -> Context:
    try:
        cap = cv2.VideoCapture(ctx.input.file_path)

        if not cap.isOpened():
            ctx.video.error = f"Unable to open file: {ctx.input.file_path}"
            return ctx

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        ctx.video.frames = frames
        ctx.video.frame_count = len(frames)
        ctx.video.fps = fps
        ctx.video.width = width
        ctx.video.height = height
        ctx.video.duration_sec = len(frames) / fps if fps > 0 else 0.0

    except Exception as e:
        ctx.video.error = str(e)

    return ctx
