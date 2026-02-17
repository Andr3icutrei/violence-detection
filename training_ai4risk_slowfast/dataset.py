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
                 slow_frames=8, fast_frames=32, temporal_stride=2,
                 slowfast_alpha=4, slowfast_beta=0.125,
                 split_ratio=0.8, training=True, augment=True,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42, use_crop=False):

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
        self.use_crop = use_crop

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

        random.seed(self.seed)
        base_path = Path(self.base_path)

        all_dirs = self.non_violence_dirs + self.violence_dirs

        for dir_name in all_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                random.shuffle(dataset_videos)

                split_idx = int(len(dataset_videos) * self.split_ratio)
                if self.training:
                    selected_videos = dataset_videos[:split_idx]
                else:
                    selected_videos = dataset_videos[split_idx:]

                label = int(dir_name)

                all_videos.extend(selected_videos)
                all_labels.extend([label] * len(selected_videos))

        combined = list(zip(all_videos, all_labels))
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

    def _transform_video_consistent(self, frames):
        do_flip = False
        brightness_factor = 1.0
        contrast_factor = 1.0
        do_random_crop = False
        crop_top, crop_left = 0, 0

        h_img, w_img = frames[0].shape[:2]

        if self.augment:
            if random.random() > 0.5:
                do_flip = True

            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)

            if self.use_crop and (h_img > self.crop_size and w_img > self.crop_size):
                do_random_crop = True
                crop_top = random.randint(0, h_img - self.crop_size)
                crop_left = random.randint(0, w_img - self.crop_size)

        processed_frames = []

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0

            if do_flip:
                frame = np.fliplr(frame).copy()

            if brightness_factor != 1.0 or contrast_factor != 1.0:
                frame = frame * brightness_factor
                mean_val = frame.mean()
                frame = (frame - mean_val) * contrast_factor + mean_val
                frame = np.clip(frame, 0, 1)

            if do_random_crop:
                frame = frame[crop_top:crop_top + self.crop_size, crop_left:crop_left + self.crop_size]
            else:
                if self.use_crop:
                    start_y = max(0, (h_img - self.crop_size) // 2)
                    start_x = max(0, (w_img - self.crop_size) // 2)
                    end_y = min(h_img, start_y + self.crop_size)
                    end_x = min(w_img, start_x + self.crop_size)
                    frame = frame[start_y:end_y, start_x:end_x]

                    if frame.shape[0] != self.crop_size or frame.shape[1] != self.crop_size:
                        frame = cv2.resize(frame, (self.crop_size, self.crop_size))
                else:
                    frame = cv2.resize(frame, (self.crop_size, self.crop_size))

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

            slow_sequence = self._transform_video_consistent(slow_frames)
            fast_sequence = self._transform_video_consistent(fast_frames)

            slow_tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
            fast_tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)

            slow_tensor = (slow_tensor - self.mean) / self.std
            fast_tensor = (fast_tensor - self.mean) / self.std

            label = torch.LongTensor([label])[0]

            return [slow_tensor, fast_tensor], label

        except Exception as e:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)