import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os
from typing import Dict, List, Optional, Tuple, Union
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'

class SlowFastVideoDataset(Dataset):
    """Load and preprocess SlowFast video samples for training or validation."""

    def __init__(self, violence_path: Union[str, Path, dict], non_violence_path: Union[str, Path, dict], slow_frames: int=8, fast_frames: int=32, temporal_stride: int=1, slowfast_alpha: int=4, slowfast_beta: float=0.125, split_ratio: float=0.8, training: bool=True, augment: bool=True, mean: List[float]=[0.45, 0.45, 0.45], std: List[float]=[0.225, 0.225, 0.225], crop_size: int=224, seed: int=42, use_crop: bool=False) -> None:
        """Initialize the object and its runtime state."""
        self._split_rng: random.Random = random.Random(seed)
        self.slow_frames: int = slow_frames
        self.fast_frames: int = fast_frames
        self.temporal_stride: int = temporal_stride
        self.slowfast_alpha: int = slowfast_alpha
        self.slowfast_beta: float = slowfast_beta
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

    def _sample_slowfast_frames(self, frames: List[np.ndarray]) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Sample aligned slow and fast pathway frames from a video."""
        total_frames: int = len(frames)
        if total_frames == 0:
            return (None, None)
        fast_seq_len: int = self.fast_frames * self.temporal_stride
        slow_seq_len: int = self.slow_frames * self.temporal_stride * self.slowfast_alpha
        required_len: int = max(fast_seq_len, slow_seq_len)
        indices: List[int]
        if total_frames < required_len:
            indices = list(range(total_frames))
            last_idx: int = total_frames - 1
            while len(indices) < required_len:
                indices.append(last_idx)
        else:
            start_idx: int
            if self.training:
                start_idx = random.randint(0, total_frames - required_len)
            else:
                start_idx = (total_frames - required_len) // 2
            indices = list(range(start_idx, start_idx + required_len))
        slow_stride: int = self.temporal_stride * self.slowfast_alpha
        slow_indices: List[int] = indices[::slow_stride][:self.slow_frames]
        while len(slow_indices) < self.slow_frames:
            slow_indices.append(slow_indices[-1])
        fast_indices: List[int] = indices[::self.temporal_stride][:self.fast_frames]
        while len(fast_indices) < self.fast_frames:
            fast_indices.append(fast_indices[-1])
        slow_frames: List[np.ndarray] = [frames[i] for i in slow_indices]
        fast_frames: List[np.ndarray] = [frames[i] for i in fast_indices]
        return (slow_frames, fast_frames)

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

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return one preprocessed video sample and its label."""
        try:
            video_path: Path = self.video_paths[idx]
            label_val: int = self.labels[idx]
            frames: List[np.ndarray] = self._extract_frames(video_path)
            slow_frames, fast_frames = self._sample_slowfast_frames(frames)
            if slow_frames is None or fast_frames is None:
                new_idx: int = (idx + 1) % len(self)
                return self.__getitem__(new_idx)
            if len(slow_frames) != self.slow_frames or len(fast_frames) != self.fast_frames:
                new_idx: int = (idx + 1) % len(self)
                return self.__getitem__(new_idx)
            augmentation_params: dict = self._sample_augmentation_params()
            slow_sequence: np.ndarray = self._apply_transform(slow_frames, augmentation_params)
            fast_sequence: np.ndarray = self._apply_transform(fast_frames, augmentation_params)
            slow_tensor: torch.Tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
            fast_tensor: torch.Tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)
            slow_tensor = (slow_tensor - self.mean) / self.std
            fast_tensor = (fast_tensor - self.mean) / self.std
            label_tensor: torch.Tensor = torch.LongTensor([label_val])[0]
            return ([slow_tensor, fast_tensor], label_tensor)
        except Exception as e:
            new_idx: int = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)