import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import torch.nn.functional as F
import time
import os

# Suppress FFmpeg warnings (some videos have invalid AMR-WB audio tracks)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class SmartCropDataset(Dataset):
    def __init__(self, violence_path, non_violence_path, teacher_model_path, config,
                 n_frames=16, split_ratio=0.75, training=True,
                 smart_crop_prob=0.8, threshold=0.6):

        self.n_frames = n_frames
        self.split_ratio = split_ratio
        self.training = training
        self.smart_crop_prob = smart_crop_prob
        self.threshold = threshold
        self.config = config

        self.mean = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)

        if isinstance(violence_path, (list, tuple)):
            self.violence_paths = [Path(p) for p in violence_path]
            self.non_violence_paths = [Path(p) for p in non_violence_path]
        else:
            self.violence_paths = [Path(violence_path)]
            self.non_violence_paths = [Path(non_violence_path)]

        self.video_paths, self.labels = self._load_video_paths()

        self.teacher_model = self._load_teacher_model(teacher_model_path)

        self.smart_crop_success = 0
        self.smart_crop_attempts = 0
        self.smart_crop_failures = 0
        self.random_crop_count = 0

    def _load_teacher_model(self, model_path):
        from teacher_model import R3D18Violence

        device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Loading teacher model from {model_path} to {device}")
        model = R3D18Violence(num_classes=2, pretrained=False).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        print(f"Teacher model loaded successfully on {device}")
        return model

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

    def _get_smart_crop_bbox(self, frames):
        device = next(self.teacher_model.parameters()).device

        if len(frames) < self.n_frames:
            return None, "not_enough_frames"

        start_idx = len(frames) // 2 - self.n_frames // 2
        sampled_frames = frames[start_idx:start_idx + self.n_frames]

        original_h, original_w = sampled_frames[0].shape[:2]

        processed_frames = []
        for frame in sampled_frames:
            frame_normalized = frame.astype(np.float32) / 255.0
            frame_resized = cv2.resize(frame_normalized, (112, 112))
            processed_frames.append(frame_resized)

        sequence = np.stack(processed_frames, axis=0)
        sequence_tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
        sequence_tensor = (sequence_tensor - self.mean) / self.std

        input_tensor = sequence_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        try:
            with torch.enable_grad():
                output = self.teacher_model(input_tensor, return_cam=True)
                pred_class = output.argmax(dim=1).item()

                self.teacher_model.zero_grad()
                output[0, pred_class].backward()

                cam_2d = self.teacher_model.get_spatial_cam_plus_plus(pred_class)
        except Exception as e:
            return None, f"cam_error: {str(e)}"

        if cam_2d is None:
            return None, "cam_none"

        heatmap = cam_2d[0].cpu().numpy()
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))

        binary_mask = (heatmap_resized > self.threshold).astype(np.uint8)

        if binary_mask.sum() == 0:
            return None, "empty_mask"

        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        if not (rows.any() and cols.any()):
            return None, "invalid_mask"

        top = int(np.argmax(rows))
        bottom = int(len(rows) - np.argmax(rows[::-1]))
        left = int(np.argmax(cols))
        right = int(len(cols) - np.argmax(cols[::-1]))

        center_row = (top + bottom) // 2
        center_col = (left + right) // 2

        box_h = bottom - top
        box_w = right - left

        crop_h = int(box_h * 1.2)
        crop_w = int(box_w * 1.2)

        crop_h = max(crop_h, 112)
        crop_w = max(crop_w, 112)

        crop_h = min(crop_h, original_h)
        crop_w = min(crop_w, original_w)

        crop_top = max(0, min(original_h - crop_h, center_row - crop_h // 2))
        crop_left = max(0, min(original_w - crop_w, center_col - crop_w // 2))

        bbox = {
            'top': crop_top,
            'left': crop_left,
            'height': crop_h,
            'width': crop_w
        }

        return bbox, "success"

    def _apply_crop(self, frame, bbox):
        if bbox is None:
            h, w = frame.shape[:2]
            max_crop = int(min(h, w) * 0.2)
            if max_crop > 0:
                crop = random.randint(0, max_crop)
                return frame[crop:h - crop, crop:w - crop]
            return frame

        top = bbox['top']
        left = bbox['left']
        height = bbox['height']
        width = bbox['width']

        return frame[top:top + height, left:left + width]

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

        if self.training:
            if random.random() > 0.5:
                frame = np.fliplr(frame).copy()

            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                frame = np.clip(frame * brightness_factor, 0, 1)

            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                mean_val = frame.mean()
                frame = np.clip((frame - mean_val) * contrast_factor + mean_val, 0, 1)

        frame = cv2.resize(frame, target_size)
        return frame

    def get_statistics(self):
        total_attempts = self.smart_crop_attempts + self.random_crop_count
        if total_attempts == 0:
            return "No samples processed yet"

        smart_crop_rate = (self.smart_crop_success / total_attempts * 100) if total_attempts > 0 else 0

        return (f"Smart Crop Statistics:\n"
                f"  Total samples: {total_attempts}\n"
                f"  Smart crop attempts: {self.smart_crop_attempts} ({self.smart_crop_attempts / total_attempts * 100:.1f}%)\n"
                f"  Smart crop successes: {self.smart_crop_success} ({smart_crop_rate:.1f}%)\n"
                f"  Smart crop failures: {self.smart_crop_failures}\n"
                f"  Random crops: {self.random_crop_count} ({self.random_crop_count / total_attempts * 100:.1f}%)")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)

        if len(frames) < self.n_frames:
            return self.__getitem__((idx + 1) % len(self))

        bbox = None
        use_smart_crop = self.training and random.random() < self.smart_crop_prob

        if use_smart_crop:
            self.smart_crop_attempts += 1
            start_time = time.time()
            bbox, status = self._get_smart_crop_bbox(frames)
            elapsed = time.time() - start_time

            if bbox is not None:
                self.smart_crop_success += 1
            else:
                self.smart_crop_failures += 1
        else:
            self.random_crop_count += 1

        if bbox is not None:
            frames = [self._apply_crop(f, bbox) for f in frames]

        sampled_frames = self._sample_frames(frames)

        if sampled_frames is None or len(sampled_frames) != self.n_frames:
            return self.__getitem__((idx + 1) % len(self))

        processed_frames = [self._preprocess_frame(frame) for frame in sampled_frames]
        sequence = np.stack(processed_frames, axis=0)

        sequence = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
        sequence = (sequence - self.mean) / self.std

        label = torch.LongTensor([label])[0]

        return sequence, label