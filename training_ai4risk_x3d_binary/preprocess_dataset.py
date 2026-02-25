import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

INPUT_ROOT = Path(r"../../Datasets/AI4RiSK")
OUTPUT_ROOT = Path(r"../../Datasets/AI4RiSK_CROPPED_SR_V2")
TARGET_SIZE = (224, 224)
MIN_CROP_SIZE = (140, 140)
CONFIDENCE_THRESHOLD = 0.25
PADDING_PIXELS = 20


def get_motion_bbox(frames, threshold=25):
    if len(frames) < 2:
        return None

    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    found_motion = False

    for i in range(1, len(frames), 3):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(first_gray, gray)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < 150:
                continue
            found_motion = True
            (x, y, w, h) = cv2.boundingRect(c)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

    if not found_motion:
        return None

    h_img, w_img = frames[0].shape[:2]
    x1 = int(max(0, min_x - PADDING_PIXELS))
    y1 = int(max(0, min_y - PADDING_PIXELS))
    x2 = int(min(w_img, max_x + PADDING_PIXELS))
    y2 = int(min(h_img, max_y + PADDING_PIXELS))

    return (x1, y1, x2 - x1, y2 - y1)


def get_person_bbox_yolo(frames, model):
    if not frames:
        return None

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    found_person = False

    step = max(1, len(frames) // 10)
    for i in range(0, len(frames), step):
        results = model.predict(frames[i], classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                found_person = True
                coords = boxes.xyxy.cpu().numpy()
                for box in coords:
                    x1, y1, x2, y2 = box
                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)

    if not found_person:
        return None

    h_img, w_img = frames[0].shape[:2]
    x1 = int(max(0, min_x - PADDING_PIXELS))
    y1 = int(max(0, min_y - PADDING_PIXELS))
    x2 = int(min(w_img, max_x + PADDING_PIXELS))
    y2 = int(min(h_img, max_y + PADDING_PIXELS))

    return (x1, y1, x2 - x1, y2 - y1)


def get_smart_bbox(frames, model):
    h_img, w_img = frames[0].shape[:2]

    bbox_yolo = get_person_bbox_yolo(frames, model)
    bbox_motion = get_motion_bbox(frames)

    final_box = (0, 0, w_img, h_img)

    if bbox_yolo is not None:
        final_box = bbox_yolo
    elif bbox_motion is not None:
        final_box = bbox_motion

    x, y, w, h = final_box
    cx = x + w // 2
    cy = y + h // 2

    target_w = max(w, MIN_CROP_SIZE[0])
    target_h = max(h, MIN_CROP_SIZE[1])

    x1 = max(0, cx - target_w // 2)
    y1 = max(0, cy - target_h // 2)
    x2 = min(w_img, x1 + target_w)
    y2 = min(h_img, y1 + target_h)

    if x2 == w_img:
        x1 = max(0, x2 - target_w)
    if y2 == h_img:
        y1 = max(0, y2 - target_h)

    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def process_frame(frame, bbox, target_size):
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
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_LANCZOS4)

    return resized


def main():
    yolo_model = YOLO("yolov8m.pt")

    input_path_obj = Path(INPUT_ROOT)
    all_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mpg']:
        all_files.extend(input_path_obj.rglob(ext))

    count = 0
    for file_p in all_files:
        try:
            rel_path = file_p.relative_to(input_path_obj)
            out_p = Path(OUTPUT_ROOT) / rel_path
            out_p.parent.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(file_p))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) < 5:
                continue

            bbox = get_smart_bbox(frames, yolo_model)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_p), fourcc, 25.0, TARGET_SIZE)

            for frame in frames:
                processed = process_frame(frame, bbox, TARGET_SIZE)
                out.write(processed)
            out.release()

            count += 1
            if count % 50 == 0:
                print(f"Processed: {count}/{len(all_files)}")

        except Exception as e:
            print(f"Error {file_p}: {e}")


if __name__ == '__main__':
    main()