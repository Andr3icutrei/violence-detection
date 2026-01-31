import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os

# Suppress FFmpeg warnings (some videos have invalid AMR-WB audio tracks)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class SlowFastVideoDataset(Dataset):
    def __init__(self, violence_path, non_violence_path,
                 num_frames_slow=8, num_frames_fast=32,
                 alpha=4, tau_slow=16, tau_fast=2,
                 split_ratio=0.75, training=True, augment=True,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42, use_crop=True):

        self.num_frames_slow = num_frames_slow
        self.num_frames_fast = num_frames_fast
        self.alpha = alpha
        self.tau_slow = tau_slow
        self.tau_fast = tau_fast
        self.split_ratio = split_ratio
        self.training = training
        self.augment = augment and training
        self.crop_size = crop_size
        self.seed = seed
        self.use_crop = use_crop

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type = 'multiclass'
            self.base_path = violence_path['path']
            self.violence_dirs = violence_path['violence_dirs']
            self.non_violence_dirs = violence_path['non_violence_dirs']
            self.violence_paths = None
            self.non_violence_paths = None
            self.is_mix = False
        elif isinstance(violence_path, (list, tuple)):
            self.dataset_type = 'standard'
            self.violence_paths = [Path(p) for p in violence_path]
            self.non_violence_paths = [Path(p) for p in non_violence_path]
            self.is_mix = True
        else:
            self.dataset_type = 'standard'
            self.violence_paths = [Path(violence_path)]
            self.non_violence_paths = [Path(non_violence_path)]
            self.is_mix = False

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        violent_videos = []
        non_violent_videos = []

        random.seed(self.seed)

        if self.dataset_type == 'multiclass':
            base_path = Path(self.base_path)

            for dir_name in self.violence_dirs:
                dir_path = base_path / dir_name
                if dir_path.exists():
                    dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                    random.shuffle(dataset_videos)
                    split_idx = int(len(dataset_videos) * self.split_ratio)

                    if self.training:
                        violent_videos.extend(dataset_videos[:split_idx])
                    else:
                        violent_videos.extend(dataset_videos[split_idx:])

            for dir_name in self.non_violence_dirs:
                dir_path = base_path / dir_name
                if dir_path.exists():
                    dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                    random.shuffle(dataset_videos)
                    split_idx = int(len(dataset_videos) * self.split_ratio)

                    if self.training:
                        non_violent_videos.extend(dataset_videos[:split_idx])
                    else:
                        non_violent_videos.extend(dataset_videos[split_idx:])
        else:
            for violence_path in self.violence_paths:
                dataset_videos = sorted([f for f in violence_path.rglob('*') if f.is_file()])
                random.shuffle(dataset_videos)
                split_idx = int(len(dataset_videos) * self.split_ratio)

                if self.training:
                    violent_videos.extend(dataset_videos[:split_idx])
                else:
                    violent_videos.extend(dataset_videos[split_idx:])

            for non_violence_path in self.non_violence_paths:
                dataset_videos = sorted([f for f in non_violence_path.rglob('*') if f.is_file()])
                random.shuffle(dataset_videos)
                split_idx = int(len(dataset_videos) * self.split_ratio)

                if self.training:
                    non_violent_videos.extend(dataset_videos[:split_idx])
                else:
                    non_violent_videos.extend(dataset_videos[split_idx:])

        videos = violent_videos + non_violent_videos
        labels = [1] * len(violent_videos) + [0] * len(non_violent_videos)

        combined = list(zip(videos, labels))
        random.shuffle(combined)
        videos, labels = zip(*combined) if combined else ([], [])

        random.seed()

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

    def _sample_frames_slowfast(self, frames):
        total_frames = len(frames)

        if total_frames == 0:
            return None, None

        temporal_window = self.num_frames_fast * self.tau_fast

        if total_frames < temporal_window:
            indices = [i % total_frames for i in range(temporal_window)]
        else:
            if self.training:
                start_idx = random.randint(0, total_frames - temporal_window)
            else:
                start_idx = (total_frames - temporal_window) // 2

            indices = list(range(start_idx, start_idx + temporal_window))

        fast_indices = indices[::self.tau_fast][:self.num_frames_fast]

        slow_indices = fast_indices[::self.alpha][:self.num_frames_slow]

        if len(fast_indices) < self.num_frames_fast:
            padding_needed = self.num_frames_fast - len(fast_indices)
            for i in range(padding_needed):
                fast_indices.append(fast_indices[i % len(fast_indices)])

        if len(slow_indices) < self.num_frames_slow:
            padding_needed = self.num_frames_slow - len(slow_indices)
            for i in range(padding_needed):
                slow_indices.append(slow_indices[i % len(slow_indices)])

        slow_frames = [frames[i] for i in slow_indices]
        fast_frames = [frames[i] for i in fast_indices]

        return slow_frames, fast_frames

    def _preprocess_frame(self, frame, target_size=256):
        frame = frame.astype(np.float32) / 255.0

        if self.use_crop:
            h, w = frame.shape[:2]
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            if self.augment:
                if random.random() > 0.5:
                    frame = np.fliplr(frame).copy()

                if random.random() > 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    frame = np.clip(frame * brightness_factor, 0, 1)

                if random.random() > 0.5:
                    contrast_factor = random.uniform(0.8, 1.2)
                    mean_val = frame.mean()
                    frame = np.clip((frame - mean_val) * contrast_factor + mean_val, 0, 1)

                if random.random() > 0.5:
                    hue_factor = random.uniform(-0.1, 0.1)
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
                    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

                if random.random() > 0.7:
                    noise = np.random.normal(0, 0.02, frame.shape)
                    frame = np.clip(frame + noise, 0, 1)

            h, w = frame.shape[:2]
            if self.training and self.augment:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2

            frame = frame[top:top + self.crop_size, left:left + self.crop_size]
        else:
            if self.augment:
                if random.random() > 0.5:
                    frame = np.fliplr(frame).copy()

                if random.random() > 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    frame = np.clip(frame * brightness_factor, 0, 1)

                if random.random() > 0.5:
                    contrast_factor = random.uniform(0.8, 1.2)
                    mean_val = frame.mean()
                    frame = np.clip((frame - mean_val) * contrast_factor + mean_val, 0, 1)

                if random.random() > 0.5:
                    hue_factor = random.uniform(-0.1, 0.1)
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
                    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

                if random.random() > 0.7:
                    noise = np.random.normal(0, 0.02, frame.shape)
                    frame = np.clip(frame + noise, 0, 1)

            frame = cv2.resize(frame, (self.crop_size, self.crop_size))

        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        slow_frames, fast_frames = self._sample_frames_slowfast(frames)

        if slow_frames is None or fast_frames is None:
            return self.__getitem__((idx + 1) % len(self))

        if len(slow_frames) != self.num_frames_slow or len(fast_frames) != self.num_frames_fast:
            return self.__getitem__((idx + 1) % len(self))

        processed_slow = [self._preprocess_frame(frame) for frame in slow_frames]
        processed_fast = [self._preprocess_frame(frame) for frame in fast_frames]

        slow_sequence = np.stack(processed_slow, axis=0)
        fast_sequence = np.stack(processed_fast, axis=0)

        slow_tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
        fast_tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)

        slow_tensor = (slow_tensor - self.mean) / self.std
        fast_tensor = (fast_tensor - self.mean) / self.std

        label = torch.LongTensor([label])[0]

        return [slow_tensor, fast_tensor], label