import cv2
import numpy as np
from ultralytics import YOLO


def preprocess_video_for_inference(video_path, yolo_model, config):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) < 2: return []

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    for frame in frames:
        results = yolo_model(frame, verbose=False, classes=[0])
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

    if min_x != float('inf'):
        h, w = frames[0].shape[:2]
        padding = 20
        min_x = max(0, int(min_x) - padding)
        min_y = max(0, int(min_y) - padding)
        max_x = min(w, int(max_x) + padding)
        max_y = min(h, int(max_y) + padding)

        cropped_frames = []
        for frame in frames:
            cropped_frames.append(frame[min_y:max_y, min_x:max_x])
        frames = cropped_frames

    processed_frames = []
    for f in frames:
        resized = cv2.resize(f, config.TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        processed_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    return processed_frames
