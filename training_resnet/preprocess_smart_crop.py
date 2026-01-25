import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from model import R3D18Violence
from config import R3DTransferConfig


class SmartCropPreprocessor:
    def __init__(self, model_path, config, threshold=0.6):
        self.config = config
        self.threshold = threshold
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = R3D18Violence(num_classes=2, pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.mean = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)

    def generate_heatmap(self, sequence, target_class):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)

        input_tensor = sequence.unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        with torch.enable_grad():
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

    def process_video(self, video_path, label, n_frames=16):
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) < n_frames:
            return None, None

        total_frames = len(frames)
        start_idx = max(0, (total_frames - n_frames) // 2)
        sampled_frames = frames[start_idx:start_idx + n_frames]

        if len(sampled_frames) != n_frames:
            return None, None

        temp_sequence = []
        for frame in sampled_frames:
            frame_normalized = frame.astype(np.float32) / 255.0
            temp_sequence.append(frame_normalized)

        temp_sequence = np.stack(temp_sequence, axis=0)
        temp_tensor = torch.FloatTensor(temp_sequence).permute(3, 0, 1, 2)
        temp_tensor = (temp_tensor - self.mean) / self.std

        heatmap = self.generate_heatmap(temp_tensor, label)

        if heatmap is None:
            return None, None

        crop_coords = self.get_crop_coordinates(heatmap, sampled_frames[0].shape)

        return crop_coords, sampled_frames

    def save_cropped_video(self, frames, crop_coords, output_path, target_size=(112, 112)):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, target_size)

        for frame in frames:
            if crop_coords is not None:
                top, left, height, width = crop_coords
                cropped = frame[top:top + height, left:left + width]
                if cropped.shape[:2] == target_size:
                    frame_to_write = cropped
                else:
                    frame_to_write = cv2.resize(cropped, target_size)
            else:
                frame_to_write = cv2.resize(frame, target_size)

            frame_bgr = cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()


def preprocess_dataset(model_path, config, output_base_path, smart_crop_prob=0.8):
    output_base_path = Path(output_base_path)

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = [config.VIOLENCE_PATH]
        non_violence_paths = [config.NON_VIOLENCE_PATH]

    preprocessor = SmartCropPreprocessor(model_path, config)

    metadata = {
        'model_path': str(model_path),
        'smart_crop_prob': smart_crop_prob,
        'threshold': preprocessor.threshold,
        'processed_videos': {'violence': [], 'non_violence': []}
    }

    for label, paths in [(1, violence_paths), (0, non_violence_paths)]:
        label_name = 'Violence' if label == 1 else 'NonViolence'

        all_videos = []
        for path in paths:
            dataset_videos = sorted([f for f in path.rglob('*') if f.is_file()])
            all_videos.extend(dataset_videos)

        output_dir = output_base_path / label_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in tqdm(all_videos, desc=f"Processing {label_name}"):
            frames = None
            crop_coords = None
            use_smart_crop = np.random.random() < smart_crop_prob

            if use_smart_crop:
                crop_coords, frames = preprocessor.process_video(video_path, label)

                if crop_coords is None or frames is None:
                    use_smart_crop = False

            if not use_smart_crop:
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cap.release()

                if len(frames) < config.N_FRAMES:
                    continue

                crop_coords = None

            output_filename = f"{video_path.stem}_cropped.mp4"
            output_path = output_dir / output_filename

            preprocessor.save_cropped_video(frames, crop_coords, output_path)

            # Convert crop_coords to Python int for JSON serialization
            crop_coords_serializable = None
            if crop_coords:
                crop_coords_serializable = tuple(int(x) for x in crop_coords)

            metadata['processed_videos']['violence' if label == 1 else 'non_violence'].append({
                'original': str(video_path),
                'processed': str(output_path),
                'smart_crop_applied': use_smart_crop,
                'crop_coords': crop_coords_serializable
            })

    metadata_path = output_base_path / 'preprocessing_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    config = R3DTransferConfig(dataset_name='Mix')

    model_path = "checkpoints_r3d18_mix/r3d18_violence_mix_best.pth"
    output_path = "../../Datasets/Mix_SmartCropped"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    preprocess_dataset(
        model_path=model_path,
        config=config,
        output_base_path=output_path,
        smart_crop_prob=0.8
    )


if __name__ == "__main__":
    main()