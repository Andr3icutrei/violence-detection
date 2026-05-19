import cv2
import numpy as np

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OnnxInputSpec:
    name: str
    layout: str
    num_frames: int
    height: int
    width: int


def _select_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames < num_frames:
        indices = list(range(total_frames))
        last_idx = total_frames - 1
        while len(indices) < num_frames:
            indices.append(last_idx)
        return indices
    start_idx = (total_frames - num_frames) // 2
    return list(range(start_idx, start_idx + num_frames))


def _compute_union_bbox_from_samples(
    frames: list[np.ndarray],
    yolo_model: Any,
    confidence: float,
    padding_pixels: int,
) -> tuple[int, int, int, int] | None:
    if not frames:
        return None

    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    found_person = False

    step = max(1, len(frames) // 10)
    for i in range(0, len(frames), step):
        results = yolo_model.predict(frames[i], classes=[0], conf=confidence, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            coords = boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in coords:
                found_person = True
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

    if not found_person:
        return None

    h_img, w_img = frames[0].shape[:2]
    x1 = int(max(0, min_x - padding_pixels))
    y1 = int(max(0, min_y - padding_pixels))
    x2 = int(min(w_img, max_x + padding_pixels))
    y2 = int(min(h_img, max_y + padding_pixels))

    return (x1, y1, x2 - x1, y2 - y1)


def _process_frame_tight_square(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    target_size: tuple[int, int],
) -> np.ndarray:
    if bbox is None:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)

    x, y, w, h = bbox
    crop = frame[y : y + h, x : x + w]

    if crop.size == 0:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)

    h_c, w_c = crop.shape[:2]
    max_dim = max(h_c, w_c)

    top = (max_dim - h_c) // 2
    bottom = max_dim - h_c - top
    left = (max_dim - w_c) // 2
    right = max_dim - w_c - left

    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded, target_size, interpolation=cv2.INTER_LANCZOS4)


def preprocess_video_for_inference(
    video_path: str,
    yolo_model: Any,
    input_specs: list[OnnxInputSpec],
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2 or not input_specs:
        return [], {}

    union_bbox = _compute_union_bbox_from_samples(
        frames,
        yolo_model,
        confidence=0.25,
        padding_pixels=0,
    )

    resized_by_name: dict[str, list[np.ndarray]] = {}
    for spec in input_specs:
        resized_frames = []
        target_size = (spec.width, spec.height)
        for frame in frames:
            resized = _process_frame_tight_square(frame, union_bbox, target_size)
            resized_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        resized_by_name[spec.name] = resized_frames

    overlay_frames = resized_by_name[input_specs[0].name]
    return overlay_frames, resized_by_name


def prepare_input_tensors(
    frames_by_name: dict[str, list[np.ndarray]],
    input_specs: list[OnnxInputSpec],
) -> dict[str, np.ndarray]:
    kinetics_mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    kinetics_std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    inputs: dict[str, np.ndarray] = {}
    for spec in input_specs:
        frames = frames_by_name.get(spec.name, [])
        if not frames:
            continue
        indices = _select_frame_indices(len(frames), spec.num_frames)
        selected = [frames[i] for i in indices]
        stacked = np.stack(selected, axis=0).astype(np.float32) / 255.0

        if spec.layout == "NCTHW":
            stacked = (stacked - kinetics_mean) / kinetics_std
            stacked = stacked.transpose(3, 0, 1, 2)
            stacked = np.expand_dims(stacked, axis=0)
        else:
            stacked = (stacked - kinetics_mean) / kinetics_std
            stacked = np.expand_dims(stacked, axis=0)

        inputs[spec.name] = stacked
    return inputs
