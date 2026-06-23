from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

FrameArray = np.ndarray
DatasetItem = tuple[torch.Tensor, torch.Tensor]
DatasetConfig = dict[str, Path | list[str] | str]


class X3DVideoDataset(Dataset[DatasetItem]):
    """Loads, samples, augments, normalizes, and labels videos for X3D training or validation."""

    def __init__(
        self,
        violence_path: DatasetConfig,
        non_violence_path: Optional[DatasetConfig],
        num_frames: int = 16,
        temporal_stride: int = 3,
        split_ratio: float = 0.8,
        training: bool = True,
        augment: bool = True,
        mean: Sequence[float] = (0.45, 0.45, 0.45),
        std: Sequence[float] = (0.225, 0.225, 0.225),
        crop_size: int = 224,
        seed: int = 42,
        use_crop: bool = False,
        use_optical_flow: bool = False,
    ) -> None:
        """Builds the video list and stores sampling, augmentation, and normalization settings."""

        self.num_frames: int = num_frames
        self.temporal_stride: int = temporal_stride
        self.split_ratio: float = split_ratio
        self.training: bool = training
        self.augment: bool = augment and training
        self.crop_size: int = crop_size
        self.seed: int = seed
        self.use_crop: bool = use_crop
        self.use_optical_flow: bool = use_optical_flow
        self.mean: torch.Tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std: torch.Tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get("type") == "multiclass":
            self.dataset_type: str = "multiclass"
            self.base_path: Path = Path(violence_path["path"]) # type: ignore
            self.violence_dirs: list[str] = list(violence_path["violence_dirs"]) # type: ignore
            self.non_violence_dirs: list[str] = list(violence_path["non_violence_dirs"]) # type: ignore
        else:
            raise ValueError("Only the AI4RiSK multiclass dataset format is supported.")

        self.video_paths: list[Path]
        self.labels: list[int]
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self) -> tuple[list[Path], list[int]]:
        """Collects video paths, applies a reproducible train/validation split, and assigns labels."""

        violent_videos: list[Path] = []
        non_violent_videos: list[Path] = []
        base_path: Path = Path(self.base_path)
        rng: random.Random = random.Random(self.seed)

        for dir_name in self.violence_dirs:
            dir_path: Path = base_path / dir_name
            if dir_path.exists():
                dataset_videos: list[Path] = sorted(file_path for file_path in dir_path.rglob("*") if file_path.is_file())
                rng.shuffle(dataset_videos)
                split_idx: int = int(len(dataset_videos) * self.split_ratio)
                selected_videos: list[Path] = dataset_videos[:split_idx] if self.training else dataset_videos[split_idx:]
                violent_videos.extend(selected_videos)

        for dir_name in self.non_violence_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted(file_path for file_path in dir_path.rglob("*") if file_path.is_file())
                rng.shuffle(dataset_videos)
                split_idx = int(len(dataset_videos) * self.split_ratio)
                selected_videos = dataset_videos[:split_idx] if self.training else dataset_videos[split_idx:]
                non_violent_videos.extend(selected_videos)

        videos: list[Path] = violent_videos + non_violent_videos
        labels: list[int] = [1] * len(violent_videos) + [0] * len(non_violent_videos)
        combined: list[tuple[Path, int]] = list(zip(videos, labels))
        rng.shuffle(combined)

        if not combined:
            return [], []

        shuffled_videos: tuple[Path, ...]
        shuffled_labels: tuple[int, ...]
        shuffled_videos, shuffled_labels = zip(*combined)
        return list(shuffled_videos), list(shuffled_labels)

    def _extract_frames(self, video_path: Path) -> list[FrameArray]:
        """Reads a video file and returns RGB frames."""

        capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        frames: list[FrameArray] = []

        while True:
            success: bool
            frame: Optional[FrameArray]
            success, frame = capture.read()
            if not success or frame is None:
                break

            rgb_frame: FrameArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        capture.release()
        return frames

    def _sample_frames_consistent(self, frames: list[FrameArray]) -> Optional[list[FrameArray]]:
        """Samples a fixed-length temporal sequence while padding short videos with the last frame."""

        total_frames: int = len(frames)
        if total_frames == 0:
            return None

        sequence_length: int = self.num_frames * self.temporal_stride

        if total_frames < sequence_length:
            indices: list[int] = list(range(total_frames))
            last_idx: int = total_frames - 1
            while len(indices) < sequence_length:
                indices.append(last_idx)
        else:
            start_idx: int = (
                random.randint(0, total_frames - sequence_length)
                if self.training
                else (total_frames - sequence_length) // 2
            )
            indices = list(range(start_idx, start_idx + sequence_length))

        indices = indices[:: self.temporal_stride]
        indices = indices[: self.num_frames]

        while len(indices) < self.num_frames:
            indices.append(indices[-1])

        selected_frames: list[FrameArray] = [frames[index] for index in indices]
        return selected_frames

    def _transform_video_consistent(self, frames: list[FrameArray]) -> FrameArray:
        """Applies temporally consistent augmentation, cropping, resizing, and scaling."""

        do_flip: bool = False
        brightness_factor: float = 1.0
        contrast_factor: float = 1.0
        do_random_crop: bool = False
        crop_top: int = 0
        crop_left: int = 0
        image_height: int
        image_width: int
        image_height, image_width = frames[0].shape[:2]

        if self.augment:
            do_flip = random.random() > 0.5

            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)

            if self.use_crop and image_height > self.crop_size and image_width > self.crop_size:
                do_random_crop = True
                crop_top = random.randint(0, image_height - self.crop_size)
                crop_left = random.randint(0, image_width - self.crop_size)

        processed_frames: list[FrameArray] = []

        for frame in frames:
            processed_frame: FrameArray = frame.astype(np.float32) / 255.0

            if do_flip:
                processed_frame = np.fliplr(processed_frame).copy()

            if brightness_factor != 1.0 or contrast_factor != 1.0:
                frame_mean: float = float(processed_frame.mean())
                processed_frame = processed_frame * brightness_factor
                processed_frame = (processed_frame - frame_mean) * contrast_factor + frame_mean
                processed_frame = np.clip(processed_frame, 0.0, 1.0)

            if do_random_crop:
                processed_frame = processed_frame[
                    crop_top : crop_top + self.crop_size,
                    crop_left : crop_left + self.crop_size,
                ]
            elif self.use_crop:
                start_y: int = max(0, (image_height - self.crop_size) // 2)
                start_x: int = max(0, (image_width - self.crop_size) // 2)
                end_y: int = min(image_height, start_y + self.crop_size)
                end_x: int = min(image_width, start_x + self.crop_size)
                processed_frame = processed_frame[start_y:end_y, start_x:end_x]

                if processed_frame.shape[0] != self.crop_size or processed_frame.shape[1] != self.crop_size:
                    processed_frame = cv2.resize(processed_frame, (self.crop_size, self.crop_size))
            else:
                processed_frame = cv2.resize(processed_frame, (self.crop_size, self.crop_size))

            processed_frames.append(processed_frame)

        sequence: FrameArray = np.stack(processed_frames, axis=0)
        return sequence

    def __len__(self) -> int:
        """Returns the number of videos in the selected split."""

        return len(self.video_paths)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Returns one normalized video tensor and its binary label."""

        try:
            video_path: Path = self.video_paths[idx]
            label_value: int = self.labels[idx]
            frames: list[FrameArray] = self._extract_frames(video_path)
            selected_frames: Optional[list[FrameArray]] = self._sample_frames_consistent(frames)

            if selected_frames is None or len(selected_frames) != self.num_frames:
                next_idx: int = (idx + 1) % len(self)
                return self.__getitem__(next_idx)

            sequence: FrameArray = self._transform_video_consistent(selected_frames)
            tensor: torch.Tensor = torch.as_tensor(sequence, dtype=torch.float32).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std
            label: torch.Tensor = torch.tensor(label_value, dtype=torch.long)

            return tensor, label

        except Exception:
            fallback_idx: int = random.randint(0, len(self) - 1)
            return self.__getitem__(fallback_idx)