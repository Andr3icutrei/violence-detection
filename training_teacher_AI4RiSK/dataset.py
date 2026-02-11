import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class X3DVideoDataset(Dataset):
    def __init__(self, violence_path, non_violence_path,
                 num_frames=16, temporal_stride=3,
                 split_ratio=0.8, training=True, augment=True,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42, use_crop=False,
                 use_optical_flow=False):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.split_ratio = split_ratio
        self.training = training
        self.augment = augment and training
        self.crop_size = crop_size
        self.seed = seed
        self.use_crop = use_crop
        self.use_optical_flow = use_optical_flow

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
        violent_videos = []
        non_violent_videos = []

        random.seed(self.seed)
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

    def _sample_frames_x3d(self, frames):
        total_frames = len(frames)
        if total_frames == 0: return None

        temporal_window = self.num_frames * self.temporal_stride

        if total_frames < temporal_window:
            indices = [i % total_frames for i in range(temporal_window)]
        else:
            if self.training:
                start_idx = random.randint(0, total_frames - temporal_window)
            else:
                start_idx = (total_frames - temporal_window) // 2
            indices = list(range(start_idx, start_idx + temporal_window))

        frame_indices = indices[::self.temporal_stride][:self.num_frames]

        if len(frame_indices) < self.num_frames:
            padding_needed = self.num_frames - len(frame_indices)
            for i in range(padding_needed):
                frame_indices.append(frame_indices[i % len(frame_indices)])

        selected_frames = [frames[i] for i in frame_indices]
        return selected_frames

    def _preprocess_frame(self, frame, target_size=256):
        frame = frame.astype(np.float32) / 255.0

        if self.use_crop:
            h, w = frame.shape[:2]
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            if self.augment:
                if random.random() > 0.5: frame = np.fliplr(frame).copy()
                if random.random() > 0.5:
                    brightness = random.uniform(0.8, 1.2)
                    frame = np.clip(frame * brightness, 0, 1)

            h, w = frame.shape[:2]
            if self.training and self.augment:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
            frame = frame[top:top + self.crop_size, left:left + self.crop_size]

        else:
            frame = cv2.resize(frame, (self.crop_size, self.crop_size))

            if self.augment:
                if random.random() > 0.5:
                    frame = np.fliplr(frame).copy()

                if random.random() > 0.5:
                    factor = random.uniform(0.7, 1.3)
                    frame = np.clip(frame * factor, 0, 1)

                if random.random() > 0.5:
                    factor = random.uniform(0.7, 1.3)
                    mean_val = frame.mean()
                    frame = np.clip((frame - mean_val) * factor + mean_val, 0, 1)

                if random.random() > 0.3:
                    noise = np.random.normal(0, 0.02, frame.shape)
                    frame = np.clip(frame + noise, 0, 1)

                if random.random() > 0.5:
                    k = random.choice([3, 5])
                    frame = cv2.GaussianBlur(frame, (k, k), 0)

        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]

            frames = self._extract_frames(video_path)
            selected_frames = self._sample_frames_x3d(frames)

            if selected_frames is None:
                return self.__getitem__((idx + 1) % len(self))

            if len(selected_frames) != self.num_frames:
                return self.__getitem__((idx + 1) % len(self))

            processed_frames = [self._preprocess_frame(frame) for frame in selected_frames]

            sequence = np.stack(processed_frames, axis=0)
            tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)

            tensor = (tensor - self.mean) / self.std
            label = torch.LongTensor([label])[0]

            return tensor, label

        except Exception:
            return self.__getitem__((idx + 1) % len(self))