import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random


class VideoSequenceDataset(Dataset):
    def __init__(self, violence_path, non_violence_path, n_frames=16,
                 split_ratio=0.75, training=True, augment=True,
                 mean=[0.43216, 0.394666, 0.37645],
                 std=[0.22803, 0.22145, 0.216989]):
        self.n_frames = n_frames
        self.split_ratio = split_ratio
        self.training = training
        self.augment = augment and training

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if isinstance(violence_path, (list, tuple)):
            self.violence_paths = [Path(p) for p in violence_path]
            self.non_violence_paths = [Path(p) for p in non_violence_path]
            self.is_mix = True
        else:
            self.violence_paths = [Path(violence_path)]
            self.non_violence_paths = [Path(non_violence_path)]
            self.is_mix = False

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        violent_videos = []
        non_violent_videos = []

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

    def _sample_frames(self, frames):
        total_frames = len(frames)
        if total_frames == 0:
            return None

        if total_frames < self.n_frames:
            indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
        else:
            if self.training:
                start_idx = random.randint(0, total_frames - self.n_frames)
                indices = list(range(start_idx, start_idx + self.n_frames))
            else:
                indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)

        sampled_frames = [frames[i] for i in indices]
        return sampled_frames

    def _preprocess_frame(self, frame, target_size=(112, 112)):
        frame = frame.astype(np.float32) / 255.0

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

            if random.random() > 0.7:
                h, w = frame.shape[:2]
                max_crop = int(min(h, w) * 0.1)
                if max_crop > 0:
                    crop = random.randint(0, max_crop)
                    frame = frame[crop:h - crop, crop:w - crop]

        frame = cv2.resize(frame, target_size)
        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        sampled_frames = self._sample_frames(frames)

        if sampled_frames is None or len(sampled_frames) != self.n_frames:
            return self.__getitem__((idx + 1) % len(self))

        processed_frames = [self._preprocess_frame(frame) for frame in sampled_frames]
        sequence = np.stack(processed_frames, axis=0)

        sequence = torch.FloatTensor(sequence).permute(3, 0, 1, 2)

        sequence = (sequence - self.mean) / self.std

        label = torch.LongTensor([label])[0]

        return sequence, label