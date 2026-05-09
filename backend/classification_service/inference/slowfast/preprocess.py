import cv2
import numpy as np
from ultralytics import YOLO


def process_frame_final(frame, bbox, target_size):
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]

    if crop.size == 0:
        return cv2.resize(frame, target_size)

    h_c, w_c = crop.shape[:2]
    max_dim = max(h_c, w_c)

    top = (max_dim - h_c) // 2
    bottom = max_dim - h_c - top
    left = (max_dim - w_c) // 2
    right = max_dim - w_c - left

    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded, target_size, interpolation=cv2.INTER_LANCZOS4)


def preprocess_video_for_inference(video_path, yolo_model_path, config):
    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        return []

    h_img, w_img = frames[0].shape[:2]

    yolo_min_x, yolo_min_y = float('inf'), float('inf')
    yolo_max_x, yolo_max_y = float('-inf'), float('-inf')
    found_person_yolo = False

    for i in range(0, len(frames), 10):
        results = model.predict(frames[i], classes=[0], conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        for result in results:
            if len(result.boxes) > 0:
                found_person_yolo = True
                coords = result.boxes.xyxy.cpu().numpy()
                for box in coords:
                    x1, y1, x2, y2 = box
                    yolo_min_x, yolo_min_y = min(yolo_min_x, x1), min(yolo_min_y, y1)
                    yolo_max_x, yolo_max_y = max(yolo_max_x, x2), max(yolo_max_y, y2)

    bbox_yolo = None
    if found_person_yolo:
        x1 = int(max(0, yolo_min_x - config.PADDING_PIXELS))
        y1 = int(max(0, yolo_min_y - config.PADDING_PIXELS))
        x2 = int(min(w_img, yolo_max_x + config.PADDING_PIXELS))
        y2 = int(min(h_img, yolo_max_y + config.PADDING_PIXELS))
        bbox_yolo = (x1, y1, x2 - x1, y2 - y1)

    first_gray = cv2.GaussianBlur(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (21, 21), 0)
    mot_min_x, mot_min_y = float('inf'), float('inf')
    mot_max_x, mot_max_y = float('-inf'), float('-inf')
    found_motion = False

    for i in range(1, len(frames), 3):
        gray = cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (21, 21), 0)
        frame_delta = cv2.absdiff(first_gray, gray)
        thresh = cv2.dilate(cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < 150:
                continue
            found_motion = True
            (x, y, w, h) = cv2.boundingRect(c)
            mot_min_x, mot_min_y = min(mot_min_x, x), min(mot_min_y, y)
            mot_max_x, mot_max_y = max(mot_max_x, x + w), max(mot_max_y, y + h)

    bbox_motion = None
    if found_motion:
        x1 = int(max(0, mot_min_x - config.PADDING_PIXELS))
        y1 = int(max(0, mot_min_y - config.PADDING_PIXELS))
        x2 = int(min(w_img, mot_max_x + config.PADDING_PIXELS))
        y2 = int(min(h_img, mot_max_y + config.PADDING_PIXELS))
        bbox_motion = (x1, y1, x2 - x1, y2 - y1)

    final_box = (0, 0, w_img, h_img)
    if bbox_yolo is not None:
        final_box = bbox_yolo
    elif bbox_motion is not None:
        final_box = bbox_motion

    x, y, w, h = final_box
    cx, cy = x + w // 2, y + h // 2

    target_w, target_h = max(w, config.MIN_CROP_SIZE[0]), max(h, config.MIN_CROP_SIZE[1])
    fx1, fy1 = max(0, cx - target_w // 2), max(0, cy - target_h // 2)
    fx2, fy2 = min(w_img, fx1 + target_w), min(h_img, fy1 + target_h)

    if fx2 == w_img: fx1 = max(0, fx2 - target_w)
    if fy2 == h_img: fy1 = max(0, fy2 - target_h)

    bbox_smart = (int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1))

    processed_frames = []
    for frame in frames:
        final_output = process_frame_final(frame, bbox_smart, config.TARGET_SIZE)
        processed_frames.append(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))

    return processed_frames