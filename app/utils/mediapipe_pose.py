import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose.Pose(model_complexity=1)

def extract(frame):
    # Convert BGR → RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_pose.process(rgb)

    if not result.pose_landmarks:
        return None

    lm = []
    for p in result.pose_landmarks.landmark:
        lm.append({"x": p.x, "y": p.y, "z": p.z, "vis": p.visibility})
    return lm
