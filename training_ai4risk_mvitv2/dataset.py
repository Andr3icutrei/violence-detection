import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os
from typing import Dict, List, Optional, Tuple, Union
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class MViTVideoDataset(Dataset):
    """Load and preprocess single-pathway video clips for MViT training or validation.

    Each sample is a tensor of shape [C, T, H, W] with ``num_frames`` frames
    (16 by default, matching MViT-B 16x4), NOT a slow/fast pair.
    """

    def __init__(self, violence_path: Union[str, Path, dict], non_violence_path: Union[str, Path, dict], num_frames: int = 16, temporal_stride: int = 4, split_ratio: float = 0.8, training: bool = True, augment: bool = True, mean: List[float] = [0.45, 0.45, 0.45], std: List[float] = [0.225, 0.225, 0.225], crop_size: int = 224, seed: int = 42, use_crop: bool = False) -> None:
        """Initialize the object and its runtime state."""
        self._split_rng: random.Random = random.Random(seed)
        self.num_frames: int = num_frames
        self.temporal_stride: int = temporal_stride
        self.split_ratio: float = split_ratio
        self.training: bool = training
        self.augment: bool = augment and training
        self.crop_size: int = crop_size
        self.seed: int = seed
        self.use_crop: bool = use_crop
        self.resize_dim: int = 256
        self.mean: torch.Tensor = torch.tensor(mean).view(3, 1, 1, 1)
        self.std: torch.Tensor = torch.tensor(std).view(3, 1, 1, 1)
        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type: str = 'multiclass'
            self.base_path: Path = violence_path['path']
            self.violence_dirs: List[str] = violence_path['violence_dirs']
            self.non_violence_dirs: List[str] = violence_path['non_violence_dirs']
        else:
            raise ValueError('Only AI4RiSK multiclass dataset is supported')
        self.video_paths: List[Path]
        self.labels: List[int]
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self) -> Tuple[List[Path], List[int]]:
        """Collect video paths and labels for the selected split."""
        all_videos: List[Path] = []
        all_labels: List[int] = []
        base_path: Path = Path(self.base_path)
        all_dirs: List[str] = self.non_violence_dirs + self.violence_dirs
        for dir_name in all_dirs:
            dir_path: Path = base_path / dir_name
            if dir_path.exists():
                dataset_videos: List[Path] = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                self._split_rng.shuffle(dataset_videos)
                split_idx: int = int(len(dataset_videos) * self.split_ratio)
                selected_videos: List[Path]
                if self.training:
                    selected_videos = dataset_videos[:split_idx]
                else:
                    selected_videos = dataset_videos[split_idx:]
                label: int = 0 if dir_name in self.non_violence_dirs else 1
                all_videos.extend(selected_videos)
                all_labels.extend([label] * len(selected_videos))
        combined: List[Tuple[Path, int]] = list(zip(all_videos, all_labels))
        self._split_rng.shuffle(combined)
        videos: Tuple[Path, ...]
        labels: Tuple[int, ...]
        videos, labels = zip(*combined) if combined else ((), ())
        return (list(videos), list(labels))

    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Read all RGB frames from a video file."""
        cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """Sample ``num_frames`` frames from a clip for the single MViT pathway.

        A window of ``num_frames * temporal_stride`` frames is selected (random
        start in training, centered in validation) and subsampled by
        ``temporal_stride``. Short clips fall back to uniform sampling with
        edge padding, so exactly ``num_frames`` frames are always returned.
        """
        total_frames: int = len(frames)
        if total_frames == 0:
            return None
        window: int = self.num_frames * self.temporal_stride
        indices: List[int]
        if total_frames >= window:
            start_idx: int
            if self.training:
                start_idx = random.randint(0, total_frames - window)
            else:
                start_idx = (total_frames - window) // 2
            indices = list(range(start_idx, start_idx + window))[::self.temporal_stride][:self.num_frames]
        else:
            # Uniformly spaced indices across the whole (short) clip.
            indices = np.linspace(0, total_frames - 1, self.num_frames).round().astype(int).tolist()
        while len(indices) < self.num_frames:
            indices.append(indices[-1])
        return [frames[i] for i in indices]

    def _sample_augmentation_params(self) -> dict:
        """Sample deterministic or random augmentation parameters."""
        margin: int = self.resize_dim - self.crop_size
        do_flip: bool
        do_color: bool
        brightness_factor: float
        contrast_factor: float
        do_blur: bool
        blur_kernel: int
        do_saturation: bool
        saturation_factor: float
        crop_top: int
        crop_left: int
        if self.augment:
            do_flip = random.random() > 0.5
            do_color = random.random() > 0.5
            brightness_factor = random.uniform(0.8, 1.2) if do_color else 1.0
            contrast_factor = random.uniform(0.8, 1.2) if do_color else 1.0
            do_blur = random.random() > 0.5
            blur_kernel = random.choice([3, 5]) if do_blur else 3
            do_saturation = random.random() > 0.5
            saturation_factor = random.uniform(0.7, 1.3) if do_saturation else 1.0
            crop_top = random.randint(0, max(0, margin))
            crop_left = random.randint(0, max(0, margin))
        else:
            do_flip = False
            brightness_factor = 1.0
            contrast_factor = 1.0
            do_blur = False
            blur_kernel = 3
            saturation_factor = 1.0
            crop_top = max(0, margin) // 2
            crop_left = max(0, margin) // 2
        return {'do_flip': do_flip, 'brightness_factor': brightness_factor, 'contrast_factor': contrast_factor, 'do_blur': do_blur, 'blur_kernel': blur_kernel, 'saturation_factor': saturation_factor, 'crop_top': crop_top, 'crop_left': crop_left}

    def _apply_transform(self, frames: List[np.ndarray], params: dict) -> np.ndarray:
        """Apply resizing, cropping, color, flip, and blur transforms."""
        do_flip: bool = params['do_flip']
        brightness_factor: float = params['brightness_factor']
        contrast_factor: float = params['contrast_factor']
        do_blur: bool = params['do_blur']
        blur_kernel: int = params['blur_kernel']
        saturation_factor: float = params['saturation_factor']
        crop_top: int = params['crop_top']
        crop_left: int = params['crop_left']
        processed_frames: List[np.ndarray] = []
        for frame in frames:
            frame_float: np.ndarray = frame.astype(np.float32) / 255.0
            frame_float = cv2.resize(frame_float, (self.resize_dim, self.resize_dim))
            frame_float = frame_float[crop_top:crop_top + self.crop_size, crop_left:crop_left + self.crop_size]
            if do_flip:
                frame_float = np.fliplr(frame_float).copy()
            if brightness_factor != 1.0 or contrast_factor != 1.0:
                frame_float = frame_float * brightness_factor
                mean_val: float = float(frame_float.mean())
                frame_float = (frame_float - mean_val) * contrast_factor + mean_val
                frame_float = np.clip(frame_float, 0, 1)
            if saturation_factor != 1.0:
                gray: np.ndarray = np.mean(frame_float, axis=2, keepdims=True)
                frame_float = gray + saturation_factor * (frame_float - gray)
                frame_float = np.clip(frame_float, 0, 1)
            if do_blur:
                frame_float = cv2.GaussianBlur(frame_float, (blur_kernel, blur_kernel), 0)
            processed_frames.append(frame_float)
        return np.stack(processed_frames, axis=0)

    def __len__(self) -> int:
        """Return the number of available video samples."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one preprocessed clip [C, T, H, W] and its label."""
        try:
            video_path: Path = self.video_paths[idx]
            label_val: int = self.labels[idx]
            frames: List[np.ndarray] = self._extract_frames(video_path)
            sampled_frames: Optional[List[np.ndarray]] = self._sample_frames(frames)
            if sampled_frames is None or len(sampled_frames) != self.num_frames:
                new_idx: int = (idx + 1) % len(self)
                return self.__getitem__(new_idx)
            augmentation_params: dict = self._sample_augmentation_params()
            sequence: np.ndarray = self._apply_transform(sampled_frames, augmentation_params)
            clip_tensor: torch.Tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)  # [C, T, H, W]
            clip_tensor = (clip_tensor - self.mean) / self.std
            label_tensor: torch.Tensor = torch.LongTensor([label_val])[0]
            return (clip_tensor, label_tensor)
        except Exception as e:
            new_idx: int = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)