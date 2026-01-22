import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import random

from model import R3D18Violence
from config import R3DTransferConfig


class SmartCropGenerator:
    def __init__(self, model_path, config, threshold=0.6):
        self.config = config
        self.threshold = threshold
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = R3D18Violence(num_classes=2, pretrained=False).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_heatmap(self, sequence, target_class):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)

        input_tensor = sequence.unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor, return_cam=True)

        self.model.zero_grad()
        output[0, target_class].backward()

        cam_2d = self.model.get_spatial_cam(target_class)

        if cam_2d is None:
            return None

        heatmap = cam_2d[0].cpu().numpy()

        return heatmap

    def get_crop_coordinates(self, heatmap, frame_shape, output_shape=(112, 112)):
        frame_height, frame_width = frame_shape[:2]
        output_height, output_width = output_shape

        heatmap_resized = cv2.resize(heatmap, (frame_width, frame_height))

        binary_mask = (heatmap_resized > self.threshold).astype(np.uint8)

        if binary_mask.sum() == 0:
            return None

        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])

        center_row = (top + bottom) // 2
        center_col = (left + right) // 2

        crop_top = max(0, center_row - output_height // 2)
        crop_left = max(0, center_col - output_width // 2)

        if crop_top + output_height > frame_height:
            crop_top = frame_height - output_height

        if crop_left + output_width > frame_width:
            crop_left = frame_width - output_width

        crop_top = max(0, crop_top)
        crop_left = max(0, crop_left)

        return (crop_top, crop_left, output_height, output_width)

    def apply_smart_crop(self, frame, crop_coords):
        if crop_coords is None:
            return None

        top, left, height, width = crop_coords
        cropped = frame[top:top + height, left:left + width]

        return cropped


class SmartCropDataset(Dataset):
    def __init__(self, violence_path, non_violence_path, model_path, config,
                 n_frames=16, split_ratio=0.75, training=True,
                 use_smart_crop=True, smart_crop_prob=0.7):
        self.violence_path = Path(violence_path)
        self.non_violence_path = Path(non_violence_path)
        self.n_frames = n_frames
        self.split_ratio = split_ratio
        self.training = training
        self.use_smart_crop = use_smart_crop and training
        self.smart_crop_prob = smart_crop_prob

        self.mean = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)

        if self.use_smart_crop:
            self.crop_generator = SmartCropGenerator(model_path, config)

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        violent_videos = sorted([f for f in self.violence_path.rglob('*') if f.is_file()])
        non_violent_videos = sorted([f for f in self.non_violence_path.rglob('*') if f.is_file()])

        violent_split_idx = int(len(violent_videos) * self.split_ratio)
        non_violent_split_idx = int(len(non_violent_videos) * self.split_ratio)

        if self.training:
            videos = violent_videos[:violent_split_idx] + non_violent_videos[:non_violent_split_idx]
            labels = [1] * len(violent_videos[:violent_split_idx]) + [0] * len(
                non_violent_videos[:non_violent_split_idx])
        else:
            videos = violent_videos[violent_split_idx:] + non_violent_videos[non_violent_split_idx:]
            labels = [1] * len(violent_videos[violent_split_idx:]) + [0] * len(
                non_violent_videos[non_violent_split_idx:])

        combined = list(zip(videos, labels))
        random.shuffle(combined)
        videos, labels = zip(*combined)

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

    def _preprocess_with_smart_crop(self, frames, label, target_size=(112, 112)):
        processed = []

        temp_sequence = []
        for frame in frames:
            frame_normalized = frame.astype(np.float32) / 255.0
            temp_sequence.append(frame_normalized)

        temp_sequence = np.stack(temp_sequence, axis=0)
        temp_tensor = torch.FloatTensor(temp_sequence).permute(3, 0, 1, 2)
        temp_tensor = (temp_tensor - self.mean) / self.std

        heatmap = self.crop_generator.generate_heatmap(temp_tensor, label)

        if heatmap is not None and random.random() < self.smart_crop_prob:
            crop_coords = self.crop_generator.get_crop_coordinates(
                heatmap, frames[0].shape, target_size
            )
        else:
            crop_coords = None

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0

            if random.random() > 0.5:
                frame = np.fliplr(frame).copy()

            if crop_coords is not None:
                cropped = self.crop_generator.apply_smart_crop(frame, crop_coords)
                if cropped is not None and cropped.shape[:2] == target_size:
                    frame = cropped
                else:
                    frame = cv2.resize(frame, target_size)
            else:
                frame = cv2.resize(frame, target_size)

            processed.append(frame)

        return processed

    def _preprocess_standard(self, frames, target_size=(112, 112)):
        processed = []

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0

            if self.training and random.random() > 0.5:
                frame = np.fliplr(frame).copy()

            frame = cv2.resize(frame, target_size)

            processed.append(frame)

        return processed

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        sampled_frames = self._sample_frames(frames)

        if sampled_frames is None or len(sampled_frames) != self.n_frames:
            return self.__getitem__((idx + 1) % len(self))

        if self.use_smart_crop and self.training:
            processed_frames = self._preprocess_with_smart_crop(sampled_frames, label)
        else:
            processed_frames = self._preprocess_standard(sampled_frames)

        sequence = np.stack(processed_frames, axis=0)
        sequence = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
        sequence = (sequence - self.mean) / self.std

        label = torch.LongTensor([label])[0]

        return sequence, label


def test_smart_crop():
    config = R3DTransferConfig()
    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    dataset = SmartCropDataset(
        violence_path=config.VIOLENCE_PATH,
        non_violence_path=config.NON_VIOLENCE_PATH,
        model_path=model_path,
        config=config,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=True,
        use_smart_crop=True,
        smart_crop_prob=0.7
    )

    sequence, label = dataset[0]
    print(f"Smart crop test successful!")
    print(f"Sequence shape: {sequence.shape}")
    print(f"Label: {label.item()}")


if __name__ == "__main__":
    test_smart_crop()