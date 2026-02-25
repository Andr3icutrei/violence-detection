import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class X3DVideoDataset(Dataset):
    def __init__(self, dataset_info,
                 num_frames=16, temporal_stride=2,
                 split_ratio=0.8, split_seed=42,
                 training=True, augment=True,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, use_crop=False,
                 max_retries=10,
                 aug_flip_prob=0.5,
                 aug_color_prob=0.5,
                 aug_brightness_range=(0.75, 1.25),
                 aug_contrast_range=(0.75, 1.25),
                 aug_rotation_prob=0.3,
                 aug_rotation_max_degrees=10,
                 aug_cutout_prob=0.3,
                 aug_cutout_size_ratio=(0.1, 0.25)):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.training = training
        self.augment = augment and training
        self.crop_size = crop_size
        self.use_crop = use_crop
        self.max_retries = max_retries

        self.aug_flip_prob = aug_flip_prob
        self.aug_color_prob = aug_color_prob
        self.aug_brightness_range = aug_brightness_range
        self.aug_contrast_range = aug_contrast_range
        self.aug_rotation_prob = aug_rotation_prob
        self.aug_rotation_max_degrees = aug_rotation_max_degrees
        self.aug_cutout_prob = aug_cutout_prob
        self.aug_cutout_size_ratio = aug_cutout_size_ratio

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if not (isinstance(dataset_info, dict) and dataset_info.get('type') == 'multiclass'):
            raise ValueError("Only AI4RiSK multiclass dataset is supported")

        self.base_path = Path(dataset_info['path'])
        self.dirs = dataset_info['dirs']

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        all_videos = []
        all_labels = []

        rng = random.Random(self.split_seed)

        for dir_name in self.dirs:
            label = int(dir_name)
            dir_path = self.base_path / dir_name

            if not dir_path.exists():
                continue

            videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
            rng.shuffle(videos)

            split_idx = int(len(videos) * self.split_ratio)

            if self.training:
                split_videos = videos[:split_idx]
            else:
                split_videos = videos[split_idx:]

            all_videos.extend(split_videos)
            all_labels.extend([label] * len(split_videos))

        combined = list(zip(all_videos, all_labels))
        rng.shuffle(combined)

        if not combined:
            return [], []

        videos, labels = zip(*combined)
        return list(videos), list(labels)

    def get_class_counts(self):
        counts = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

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

        seq_len = self.num_frames * self.temporal_stride

        if total_frames < seq_len:
            indices = list(range(total_frames))
            last_idx = total_frames - 1
            while len(indices) < seq_len:
                indices.append(last_idx)
        else:
            if self.training:
                start_idx = random.randint(0, total_frames - seq_len)
            else:
                start_idx = (total_frames - seq_len) // 2

            indices = list(range(start_idx, start_idx + seq_len))

        indices = indices[::self.temporal_stride]
        indices = indices[:self.num_frames]

        while len(indices) < self.num_frames:
            indices.append(indices[-1])

        return [frames[i] for i in indices]

    def _sample_augment_params(self, h_img, w_img):
        params = {
            'do_flip': False,
            'brightness_factor': 1.0,
            'contrast_factor': 1.0,
            'rotation_angle': 0.0,
            'do_cutout': False,
            'cutout_top': 0,
            'cutout_left': 0,
            'cutout_h': 0,
            'cutout_w': 0,
            'do_random_crop': False,
            'crop_top': 0,
            'crop_left': 0,
        }

        if not self.augment:
            return params

        if random.random() < self.aug_flip_prob:
            params['do_flip'] = True

        if random.random() < self.aug_color_prob:
            params['brightness_factor'] = random.uniform(*self.aug_brightness_range)
            params['contrast_factor'] = random.uniform(*self.aug_contrast_range)

        if random.random() < self.aug_rotation_prob:
            params['rotation_angle'] = random.uniform(
                -self.aug_rotation_max_degrees,
                self.aug_rotation_max_degrees
            )

        if random.random() < self.aug_cutout_prob:
            min_ratio, max_ratio = self.aug_cutout_size_ratio
            cutout_h = int(random.uniform(min_ratio, max_ratio) * h_img)
            cutout_w = int(random.uniform(min_ratio, max_ratio) * w_img)
            cutout_top = random.randint(0, max(0, h_img - cutout_h))
            cutout_left = random.randint(0, max(0, w_img - cutout_w))
            params['do_cutout'] = True
            params['cutout_top'] = cutout_top
            params['cutout_left'] = cutout_left
            params['cutout_h'] = cutout_h
            params['cutout_w'] = cutout_w

        if self.use_crop and h_img > self.crop_size and w_img > self.crop_size:
            params['do_random_crop'] = True
            params['crop_top'] = random.randint(0, h_img - self.crop_size)
            params['crop_left'] = random.randint(0, w_img - self.crop_size)

        return params

    def _apply_rotation(self, frame, angle):
        if angle == 0.0:
            return frame
        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return rotated

    def _transform_frames(self, frames):
        h_img, w_img = frames[0].shape[:2]
        params = self._sample_augment_params(h_img, w_img)

        processed_frames = []

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0

            if params['do_flip']:
                frame = np.fliplr(frame).copy()

            if params['rotation_angle'] != 0.0:
                frame = self._apply_rotation(frame, params['rotation_angle'])

            if params['brightness_factor'] != 1.0 or params['contrast_factor'] != 1.0:
                frame = frame * params['brightness_factor']
                mean_val = frame.mean()
                frame = (frame - mean_val) * params['contrast_factor'] + mean_val
                frame = np.clip(frame, 0.0, 1.0)

            if params['do_random_crop']:
                top = params['crop_top']
                left = params['crop_left']
                frame = frame[top:top + self.crop_size, left:left + self.crop_size]
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

            if params['do_cutout']:
                top = params['cutout_top']
                left = params['cutout_left']
                ch = min(params['cutout_h'], frame.shape[0] - top)
                cw = min(params['cutout_w'], frame.shape[1] - left)
                if ch > 0 and cw > 0:
                    frame[top:top + ch, left:left + cw] = 0.45

            processed_frames.append(frame)

        return np.stack(processed_frames, axis=0)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        retries = 0
        current_idx = idx

        while retries < self.max_retries:
            try:
                video_path = self.video_paths[current_idx]
                label = self.labels[current_idx]

                frames = self._extract_frames(video_path)
                selected_frames = self._sample_frames(frames)

                if selected_frames is None or len(selected_frames) != self.num_frames:
                    raise ValueError(f"Invalid frame count for {video_path}")

                sequence = self._transform_frames(selected_frames)
                tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
                tensor = (tensor - self.mean) / self.std

                return tensor, torch.tensor(label, dtype=torch.long)

            except Exception:
                retries += 1
                current_idx = random.randint(0, len(self) - 1)

        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries} retries starting from index {idx}"
        )


class MultiViewX3DDataset(Dataset):
    def __init__(self, dataset_info,
                 num_frames=16, temporal_stride=2,
                 split_ratio=0.8, split_seed=42,
                 num_clips=5,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, use_crop=False):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.num_clips = num_clips
        self.crop_size = crop_size
        self.use_crop = use_crop

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if not (isinstance(dataset_info, dict) and dataset_info.get('type') == 'multiclass'):
            raise ValueError("Only AI4RiSK multiclass dataset is supported")

        self.base_path = Path(dataset_info['path'])
        self.dirs = dataset_info['dirs']

        self.video_paths, self.labels = self._load_video_paths(split_ratio, split_seed)

    def _load_video_paths(self, split_ratio, split_seed):
        all_videos = []
        all_labels = []

        rng = random.Random(split_seed)

        for dir_name in self.dirs:
            label = int(dir_name)
            dir_path = self.base_path / dir_name

            if not dir_path.exists():
                continue

            videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
            rng.shuffle(videos)

            split_idx = int(len(videos) * split_ratio)
            split_videos = videos[split_idx:]

            all_videos.extend(split_videos)
            all_labels.extend([label] * len(split_videos))

        return all_videos, all_labels

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

    def _extract_clips(self, frames):
        total_frames = len(frames)
        temporal_window = self.num_frames * self.temporal_stride

        if total_frames <= temporal_window:
            padded = list(range(total_frames))
            last = total_frames - 1
            while len(padded) < temporal_window:
                padded.append(last)
            return [padded]

        # Maximum number of non-redundant start positions
        max_starts = total_frames - temporal_window + 1
        # Clamp requested clips to what is actually available
        actual_clips = min(self.num_clips, max_starts)

        if actual_clips == 1:
            start_positions = [0]
        else:
            step = (total_frames - temporal_window) / (actual_clips - 1)
            start_positions = [
                int(round(i * step)) for i in range(actual_clips)
            ]

        clips = []
        for start in start_positions:
            start = min(start, total_frames - temporal_window)
            clips.append(list(range(start, start + temporal_window)))

        return clips

    def _preprocess_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0

        if self.use_crop:
            h, w = frame.shape[:2]
            scale = 256 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            h, w = frame.shape[:2]
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            frame = frame[top:top + self.crop_size, left:left + self.crop_size]
        else:
            frame = cv2.resize(frame, (self.crop_size, self.crop_size))

        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        clip_indices_list = self._extract_clips(frames)

        processed_clips = []

        for clip_indices in clip_indices_list:
            frame_indices = clip_indices[::self.temporal_stride][:self.num_frames]

            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])

            selected_frames = [frames[i] for i in frame_indices]
            processed_frames = [self._preprocess_frame(f) for f in selected_frames]

            sequence = np.stack(processed_frames, axis=0)
            tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std

            processed_clips.append(tensor)

        if not processed_clips:
            processed_clips = [torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)]

        return processed_clips, torch.tensor(label, dtype=torch.long)