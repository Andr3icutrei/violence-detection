from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


logger: logging.Logger = logging.getLogger(__name__)

INPUT_ROOT: Path = Path(r"../../Datasets/AI4RiSK")
OUTPUT_ROOT: Path = Path(r"../../Datasets/AI4RiSK_CROPPED_SR_V2")
TARGET_SIZE: tuple[int, int] = (224, 224)
MIN_CROP_SIZE: tuple[int, int] = (140, 140)
CONFIDENCE_THRESHOLD: float = 0.25
PADDING_PIXELS: int = 20
MOTION_THRESHOLD: int = 25
MOTION_FRAME_STEP: int = 3
MIN_MOTION_AREA: float = 150.0
YOLO_PERSON_CLASS_ID: int = 0
YOLO_SAMPLE_COUNT: int = 10
MIN_VIDEO_FRAMES: int = 5
VIDEO_EXTENSIONS: tuple[str, ...] = ("*.mp4", "*.avi", "*.mov", "*.mpg")
OUTPUT_FPS: float = 25.0
YOLO_MODEL_PATH: Path = Path("../training_ai4risk_x3dL_binary/yolov8m.pt")

FrameArray = np.ndarray
BoundingBox = tuple[int, int, int, int]


def get_motion_bbox(frames: list[FrameArray], threshold: int = MOTION_THRESHOLD) -> Optional[BoundingBox]:
    """Computes a bounding box around motion regions across sampled frames."""

    if len(frames) < 2:
        return None

    first_gray: FrameArray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

    min_x: float = float("inf")
    min_y: float = float("inf")
    max_x: float = float("-inf")
    max_y: float = float("-inf")
    found_motion: bool = False

    for frame_index in range(1, len(frames), MOTION_FRAME_STEP):
        frame: FrameArray = frames[frame_index]
        gray: FrameArray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        frame_delta: FrameArray = cv2.absdiff(first_gray, gray)
        threshold_frame: FrameArray = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
        contours: tuple[FrameArray, ...]
        contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area: float = float(cv2.contourArea(contour))
            if contour_area < MIN_MOTION_AREA:
                continue

            found_motion = True
            x: int
            y: int
            width: int
            height: int
            x, y, width, height = cv2.boundingRect(contour)
            min_x = min(min_x, float(x))
            min_y = min(min_y, float(y))
            max_x = max(max_x, float(x + width))
            max_y = max(max_y, float(y + height))

    if not found_motion:
        return None

    image_height: int
    image_width: int
    image_height, image_width = frames[0].shape[:2]
    x1: int = int(max(0, min_x - PADDING_PIXELS))
    y1: int = int(max(0, min_y - PADDING_PIXELS))
    x2: int = int(min(image_width, max_x + PADDING_PIXELS))
    y2: int = int(min(image_height, max_y + PADDING_PIXELS))

    return x1, y1, x2 - x1, y2 - y1


def get_person_bbox_yolo(frames: list[FrameArray], model: YOLO) -> Optional[BoundingBox]:
    """Computes a bounding box around detected people using YOLO."""

    if not frames:
        return None

    min_x: float = float("inf")
    min_y: float = float("inf")
    max_x: float = float("-inf")
    max_y: float = float("-inf")
    found_person: bool = False
    step: int = max(1, len(frames) // YOLO_SAMPLE_COUNT)

    for frame_index in range(0, len(frames), step):
        results = model.predict(
            frames[frame_index],
            classes=[YOLO_PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue

            found_person = True
            coordinates: FrameArray = boxes.xyxy.cpu().numpy()

            for box in coordinates:
                x1: float
                y1: float
                x2: float
                y2: float
                x1, y1, x2, y2 = box
                min_x = min(min_x, float(x1))
                min_y = min(min_y, float(y1))
                max_x = max(max_x, float(x2))
                max_y = max(max_y, float(y2))

    if not found_person:
        return None

    image_height: int
    image_width: int
    image_height, image_width = frames[0].shape[:2]
    padded_x1: int = int(max(0, min_x - PADDING_PIXELS))
    padded_y1: int = int(max(0, min_y - PADDING_PIXELS))
    padded_x2: int = int(min(image_width, max_x + PADDING_PIXELS))
    padded_y2: int = int(min(image_height, max_y + PADDING_PIXELS))

    return padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1


def get_smart_bbox(frames: list[FrameArray], model: YOLO) -> BoundingBox:
    """Chooses a person, motion, or full-frame bounding box and enforces a minimum crop size."""

    image_height: int
    image_width: int
    image_height, image_width = frames[0].shape[:2]

    bbox_yolo: Optional[BoundingBox] = get_person_bbox_yolo(frames, model)
    bbox_motion: Optional[BoundingBox] = get_motion_bbox(frames)
    final_box: BoundingBox = (0, 0, image_width, image_height)

    if bbox_yolo is not None:
        final_box = bbox_yolo
    elif bbox_motion is not None:
        final_box = bbox_motion

    x: int
    y: int
    width: int
    height: int
    x, y, width, height = final_box
    center_x: int = x + width // 2
    center_y: int = y + height // 2
    target_width: int = max(width, MIN_CROP_SIZE[0])
    target_height: int = max(height, MIN_CROP_SIZE[1])
    x1: int = max(0, center_x - target_width // 2)
    y1: int = max(0, center_y - target_height // 2)
    x2: int = min(image_width, x1 + target_width)
    y2: int = min(image_height, y1 + target_height)

    if x2 == image_width:
        x1 = max(0, x2 - target_width)

    if y2 == image_height:
        y1 = max(0, y2 - target_height)

    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def process_frame(frame: FrameArray, bbox: BoundingBox, target_size: tuple[int, int]) -> FrameArray:
    """Crops, square-pads, and resizes a single frame."""

    x: int
    y: int
    width: int
    height: int
    x, y, width, height = bbox
    crop: FrameArray = frame[y : y + height, x : x + width]

    if crop.size == 0:
        return cv2.resize(frame, target_size)

    crop_height: int
    crop_width: int
    crop_height, crop_width = crop.shape[:2]
    max_dim: int = max(crop_height, crop_width)
    top: int = (max_dim - crop_height) // 2
    bottom: int = max_dim - crop_height - top
    left: int = (max_dim - crop_width) // 2
    right: int = max_dim - crop_width - left
    padded: FrameArray = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    resized: FrameArray = cv2.resize(padded, target_size, interpolation=cv2.INTER_LANCZOS4)

    return resized


def read_video_frames(video_path: Path) -> list[FrameArray]:
    """Reads all frames from a video file."""

    capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    frames: list[FrameArray] = []

    while True:
        success: bool
        frame: Optional[FrameArray]
        success, frame = capture.read()
        if not success or frame is None:
            break
        frames.append(frame)

    capture.release()
    return frames


def write_processed_video(output_path: Path, frames: list[FrameArray], bbox: BoundingBox) -> None:
    """Writes processed frames to an output video file."""

    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, TARGET_SIZE)

    for frame in frames:
        processed_frame: FrameArray = process_frame(frame, bbox, TARGET_SIZE)
        writer.write(processed_frame)

    writer.release()


def main() -> None:
    """Preprocesses the AI4RiSK dataset into cropped and resized videos."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    yolo_model: YOLO = YOLO(str(YOLO_MODEL_PATH))
    input_path: Path = Path(INPUT_ROOT)
    all_files: list[Path] = []

    for extension in VIDEO_EXTENSIONS:
        all_files.extend(input_path.rglob(extension))

    processed_count: int = 0
    skipped_count: int = 0

    for file_path in all_files:
        try:
            relative_path: Path = file_path.relative_to(input_path)
            output_path: Path = Path(OUTPUT_ROOT) / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            frames: list[FrameArray] = read_video_frames(file_path)
            if len(frames) < MIN_VIDEO_FRAMES:
                skipped_count += 1
                continue

            bbox: BoundingBox = get_smart_bbox(frames, yolo_model)
            write_processed_video(output_path, frames, bbox)
            processed_count += 1

        except Exception as exc:
            skipped_count += 1
            logger.warning("Could not process %s: %s", file_path, exc)

    logger.info("Preprocessing complete. Processed=%d | skipped=%d | output=%s", processed_count, skipped_count, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
