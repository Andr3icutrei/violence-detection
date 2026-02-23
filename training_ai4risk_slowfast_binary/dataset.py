import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class SlowFastVideoDataset(Dataset):
    def __init__(self, violence_path, non_violence_path,
                 slow_frames=8, fast_frames=32, temporal_stride=1,
                 slowfast_alpha=4, slowfast_beta=0.125,
                 split_ratio=0.8, training=True, augment=True,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42,
                 use_crop=True):
        self._split_rng = random.Random(seed)

        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.temporal_stride = temporal_stride
        self.slowfast_alpha = slowfast_alpha
        self.slowfast_beta = slowfast_beta
        self.split_ratio = split_ratio
        self.training = training
        self.augment = augment and training
        self.crop_size = crop_size
        self.seed = seed
        self.resize_dim = 256

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type = 'multiclass'
            self.base_path = violence_path['path']
            self.violence_dirs = violence_path['violence_dirs']
            self.non_violence_dirs = violence_path['non_violence_dirs']
        else:
            raise ValueError("Only AI4RiSK multiclass dataset is supported")

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        all_videos = []
        all_labels = []

        base_path = Path(self.base_path)

        all_dirs = self.non_violence_dirs + self.violence_dirs

        for dir_name in all_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                self._split_rng.shuffle(dataset_videos)

                split_idx = int(len(dataset_videos) * self.split_ratio)
                if self.training:
                    selected_videos = dataset_videos[:split_idx]
                else:
                    selected_videos = dataset_videos[split_idx:]

                label = 0 if dir_name in self.non_violence_dirs else 1

                all_videos.extend(selected_videos)
                all_labels.extend([label] * len(selected_videos))

        combined = list(zip(all_videos, all_labels))
        self._split_rng.shuffle(combined)
        videos, labels = zip(*combined) if combined else ([], [])

        return list(videos), list(labels)

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_slowfast_frames(self, frames):
        total_frames = len(frames)
        if total_frames == 0:
            return None, None

        fast_seq_len = self.fast_frames * self.temporal_stride
        slow_seq_len = self.slow_frames * self.temporal_stride * self.slowfast_alpha

        required_len = max(fast_seq_len, slow_seq_len)

        if total_frames < required_len:
            indices = list(range(total_frames))
            last_idx = total_frames - 1
            while len(indices) < required_len:
                indices.append(last_idx)
        else:
            if self.training:
                start_idx = random.randint(0, total_frames - required_len)
            else:
                start_idx = (total_frames - required_len) // 2
            indices = list(range(start_idx, start_idx + required_len))

        slow_stride = self.temporal_stride * self.slowfast_alpha
        slow_indices = indices[::slow_stride][:self.slow_frames]
        while len(slow_indices) < self.slow_frames:
            slow_indices.append(slow_indices[-1])

        fast_indices = indices[::self.temporal_stride][:self.fast_frames]
        while len(fast_indices) < self.fast_frames:
            fast_indices.append(fast_indices[-1])

        slow_frames = [frames[i] for i in slow_indices]
        fast_frames = [frames[i] for i in fast_indices]

        return slow_frames, fast_frames

    def _sample_augmentation_params(self):
        margin = self.resize_dim - self.crop_size

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

        return {
            'do_flip': do_flip,
            'brightness_factor': brightness_factor,
            'contrast_factor': contrast_factor,
            'do_blur': do_blur,
            'blur_kernel': blur_kernel,
            'saturation_factor': saturation_factor,
            'crop_top': crop_top,
            'crop_left': crop_left,
        }

    def _apply_transform(self, frames, params):
        do_flip = params['do_flip']
        brightness_factor = params['brightness_factor']
        contrast_factor = params['contrast_factor']
        do_blur = params['do_blur']
        blur_kernel = params['blur_kernel']
        saturation_factor = params['saturation_factor']
        crop_top = params['crop_top']
        crop_left = params['crop_left']

        processed_frames = []

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0

            frame = cv2.resize(frame, (self.resize_dim, self.resize_dim))

            frame = frame[crop_top: crop_top + self.crop_size,
                          crop_left: crop_left + self.crop_size]

            if do_flip:
                frame = np.fliplr(frame).copy()

            if brightness_factor != 1.0 or contrast_factor != 1.0:
                frame = frame * brightness_factor
                mean_val = frame.mean()
                frame = (frame - mean_val) * contrast_factor + mean_val
                frame = np.clip(frame, 0, 1)

            if saturation_factor != 1.0:
                gray = np.mean(frame, axis=2, keepdims=True)
                frame = gray + saturation_factor * (frame - gray)
                frame = np.clip(frame, 0, 1)

            if do_blur:
                frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

            processed_frames.append(frame)

        return np.stack(processed_frames, axis=0)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]

            frames = self._extract_frames(video_path)

            slow_frames, fast_frames = self._sample_slowfast_frames(frames)

            if slow_frames is None or fast_frames is None:
                new_idx = (idx + 1) % len(self)
                return self.__getitem__(new_idx)

            if len(slow_frames) != self.slow_frames or len(fast_frames) != self.fast_frames:
                new_idx = (idx + 1) % len(self)
                return self.__getitem__(new_idx)

            augmentation_params = self._sample_augmentation_params()

            slow_sequence = self._apply_transform(slow_frames, augmentation_params)
            fast_sequence = self._apply_transform(fast_frames, augmentation_params)

            slow_tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
            fast_tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)

            slow_tensor = (slow_tensor - self.mean) / self.std
            fast_tensor = (fast_tensor - self.mean) / self.std

            label = torch.LongTensor([label])[0]

            return [slow_tensor, fast_tensor], label

        except Exception as e:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)