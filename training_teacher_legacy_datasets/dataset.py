from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

DatasetInfo: TypeAlias = dict[str, Path | list[str] | str]
PathInput: TypeAlias = str | Path | DatasetInfo | Sequence[str | Path | DatasetInfo]
FrameArray: TypeAlias = np.ndarray


class VideoSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load labeled video clips, sample fixed-length frame sequences, and preprocess them."""

    def __init__(
        self,
        violence_path: PathInput,
        non_violence_path: PathInput,
        n_frames: int = 16,
        split_ratio: float = 0.75,
        training: bool = True,
        augment: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        """Initialize video paths, normalization tensors, and the person detector."""
        self.n_frames: int = n_frames
        self.split_ratio: float = split_ratio
        self.training: bool = training
        self.augment: bool = augment and training

        normalization_mean: Sequence[float] = mean or [0.43216, 0.394666, 0.37645]
        normalization_std: Sequence[float] = std or [0.22803, 0.22145, 0.216989]
        self.mean: torch.Tensor = torch.tensor(normalization_mean).view(3, 1, 1, 1)
        self.std: torch.Tensor = torch.tensor(normalization_std).view(3, 1, 1, 1)

        self.yolo_model: YOLO = YOLO("yolov8m.pt")

        self.dataset_type: str
        self.base_path: str | Path | None = None
        self.violence_dirs: Sequence[str] | None = None
        self.non_violence_dirs: Sequence[str] | None = None
        self.violence_paths: list[Path] | None = None
        self.non_violence_paths: list[Path] | None = None
        self.is_mix: bool = False

        if isinstance(violence_path, dict) and violence_path.get("type") == "multiclass":
            self.dataset_type = "multiclass"
            self.base_path = violence_path["path"]
            self.violence_dirs = violence_path["violence_dirs"]
            self.non_violence_dirs = violence_path["non_violence_dirs"]
        elif isinstance(violence_path, (list, tuple)):
            self.dataset_type = "standard"
            self.violence_paths = [Path(path) for path in violence_path]
            self.non_violence_paths = [Path(path) for path in non_violence_path]
            self.is_mix = True
        else:
            self.dataset_type = "standard"
            self.violence_paths = [Path(violence_path)]
            self.non_violence_paths = [Path(non_violence_path)]

        self.video_paths: list[Path]
        self.labels: list[int]
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self) -> tuple[list[Path], list[int]]:
        """Build a deterministic train or validation split and return shuffled paths with labels."""
        violent_videos: list[Path] = []
        non_violent_videos: list[Path] = []

        if self.dataset_type == "multiclass":
            base_path: Path = Path(self.base_path)
            violence_dirs: Sequence[str] = self.violence_dirs or []
            non_violence_dirs: Sequence[str] = self.non_violence_dirs or []

            for directory_name in violence_dirs:
                directory_path: Path = base_path / directory_name
                if directory_path.exists():
                    dataset_videos: list[Path] = sorted(file_path for file_path in directory_path.rglob("*") if file_path.is_file())
                    random.seed(42)
                    random.shuffle(dataset_videos)
                    split_index: int = int(len(dataset_videos) * self.split_ratio)
                    violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])

            for directory_name in non_violence_dirs:
                directory_path = base_path / directory_name
                if directory_path.exists():
                    dataset_videos = sorted(file_path for file_path in directory_path.rglob("*") if file_path.is_file())
                    random.seed(42)
                    random.shuffle(dataset_videos)
                    split_index = int(len(dataset_videos) * self.split_ratio)
                    non_violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])
        else:
            violence_paths: list[Path] = self.violence_paths or []
            non_violence_paths: list[Path] = self.non_violence_paths or []

            for violence_path in violence_paths:
                dataset_videos = sorted(file_path for file_path in violence_path.rglob("*") if file_path.is_file())
                random.seed(42)
                random.shuffle(dataset_videos)
                split_index = int(len(dataset_videos) * self.split_ratio)
                violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])

            for non_violence_path in non_violence_paths:
                dataset_videos = sorted(file_path for file_path in non_violence_path.rglob("*") if file_path.is_file())
                random.seed(42)
                random.shuffle(dataset_videos)
                split_index = int(len(dataset_videos) * self.split_ratio)
                non_violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])

        videos: list[Path] = violent_videos + non_violent_videos
        labels: list[int] = [1] * len(violent_videos) + [0] * len(non_violent_videos)
        combined: list[tuple[Path, int]] = list(zip(videos, labels))

        random.seed(42)
        random.shuffle(combined)
        random.seed()

        if not combined:
            return [], []

        shuffled_videos: tuple[Path, ...]
        shuffled_labels: tuple[int, ...]
        shuffled_videos, shuffled_labels = zip(*combined)
        return list(shuffled_videos), list(shuffled_labels)

    def _extract_and_sample_frames(self, video_path: Path) -> list[FrameArray] | None:
        """Read a fixed-length frame sequence and crop around detected people when possible."""
        video_capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        total_frames: int = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[FrameArray] = []

        if total_frames == 0:
            video_capture.release()
            return None

        required_frames: int = self.n_frames
        if total_frames <= required_frames:
            start_index: int = 0
        elif self.training:
            start_index = random.randint(0, total_frames - required_frames)
        else:
            start_index = (total_frames - required_frames) // 2

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)

        for _ in range(required_frames):
            success: bool
            frame: FrameArray
            success, frame = video_capture.read()
            if not success:
                break
            rgb_frame: FrameArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            if len(frames) == self.n_frames:
                break

        video_capture.release()

        if not frames:
            return None

        while len(frames) < self.n_frames:
            frames.append(frames[-1])

        min_x: float = float("inf")
        min_y: float = float("inf")
        max_x: float = 0.0
        max_y: float = 0.0

        for frame in frames:
            detection_results = self.yolo_model(frame, verbose=False, classes=[0])
            for result in detection_results:
                boxes: np.ndarray = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1: float
                    y1: float
                    x2: float
                    y2: float
                    x1, y1, x2, y2 = box
                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)

        if min_x != float("inf"):
            frame_height: int
            frame_width: int
            frame_height, frame_width = frames[0].shape[:2]
            padding: int = 20
            left: int = max(0, int(min_x) - padding)
            top: int = max(0, int(min_y) - padding)
            right: int = min(frame_width, int(max_x) + padding)
            bottom: int = min(frame_height, int(max_y) + padding)
            frames = [frame[top:bottom, left:right] for frame in frames]

        return frames

    def _preprocess_frame(self, frame: FrameArray, target_size: tuple[int, int] = (112, 112)) -> FrameArray:
        """Normalize, optionally augment, and resize one RGB frame."""
        processed_frame: FrameArray = frame.astype(np.float32) / 255.0

        if self.augment:
            if random.random() > 0.5:
                processed_frame = np.fliplr(processed_frame).copy()

            if random.random() > 0.5:
                brightness_factor: float = random.uniform(0.8, 1.2)
                processed_frame = np.clip(processed_frame * brightness_factor, 0, 1)

            if random.random() > 0.5:
                contrast_factor: float = random.uniform(0.8, 1.2)
                mean_value: float = float(processed_frame.mean())
                processed_frame = np.clip((processed_frame - mean_value) * contrast_factor + mean_value, 0, 1)

            if random.random() > 0.5:
                hue_factor: float = random.uniform(-0.1, 0.1)
                frame_uint8: FrameArray = (processed_frame * 255).astype(np.uint8)
                hsv_frame: FrameArray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + hue_factor * 180) % 180
                processed_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

            if random.random() > 0.7:
                noise: FrameArray = np.random.normal(0, 0.02, processed_frame.shape)
                processed_frame = np.clip(processed_frame + noise, 0, 1)

            if random.random() > 0.7:
                frame_height: int
                frame_width: int
                frame_height, frame_width = processed_frame.shape[:2]
                max_crop: int = int(min(frame_height, frame_width) * 0.1)
                if max_crop > 0:
                    crop: int = random.randint(0, max_crop)
                    processed_frame = processed_frame[crop:frame_height - crop, crop:frame_width - crop]

        resized_frame: FrameArray = cv2.resize(processed_frame, target_size)
        return resized_frame

    def __len__(self) -> int:
        """Return the number of videos in the selected split."""
        return len(self.video_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one normalized clip tensor and its class label."""
        video_path: Path = self.video_paths[index]
        label_value: int = self.labels[index]
        sampled_frames: list[FrameArray] | None = self._extract_and_sample_frames(video_path)

        if sampled_frames is None or len(sampled_frames) != self.n_frames:
            next_index: int = (index + 1) % len(self)
            return self.__getitem__(next_index)

        processed_frames: list[FrameArray] = [self._preprocess_frame(frame) for frame in sampled_frames]
        sequence_array: FrameArray = np.stack(processed_frames, axis=0)
        sequence_tensor: torch.Tensor = torch.FloatTensor(sequence_array).permute(3, 0, 1, 2)
        sequence_tensor = (sequence_tensor - self.mean) / self.std
        label_tensor: torch.Tensor = torch.LongTensor([label_value])[0]

        return sequence_tensor, label_tensor