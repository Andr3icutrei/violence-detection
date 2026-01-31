import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
import json

from model import R3D18Violence
from dataset import VideoSequenceDataset
from config import R3DTransferConfig


class MultiViewVideoDataset:
    def __init__(self, violence_path, non_violence_path, n_frames=16,
                 split_ratio=0.75, training=False, num_clips=10,
                 mean=[0.43216, 0.394666, 0.37645],
                 std=[0.22803, 0.22145, 0.216989]):
        self.n_frames = n_frames
        self.split_ratio = split_ratio
        self.training = training
        self.num_clips = num_clips

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type = 'multiclass'
            self.base_path = violence_path['path']
            self.violence_dirs = violence_path['violence_dirs']
            self.non_violence_dirs = violence_path['non_violence_dirs']
            self.violence_paths = None
            self.non_violence_paths = None
        elif isinstance(violence_path, (list, tuple)):
            self.dataset_type = 'standard'
            self.violence_paths = [Path(p) for p in violence_path]
            self.non_violence_paths = [Path(p) for p in non_violence_path]
        else:
            self.dataset_type = 'standard'
            self.violence_paths = [Path(violence_path)]
            self.non_violence_paths = [Path(non_violence_path)]

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        violent_videos = []
        non_violent_videos = []

        if self.dataset_type == 'multiclass':
            base_path = Path(self.base_path)

            for dir_name in self.violence_dirs:
                dir_path = base_path / dir_name
                if dir_path.exists():
                    dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                    split_idx = int(len(dataset_videos) * self.split_ratio)

                    if self.training:
                        violent_videos.extend(dataset_videos[:split_idx])
                    else:
                        violent_videos.extend(dataset_videos[split_idx:])

            for dir_name in self.non_violence_dirs:
                dir_path = base_path / dir_name
                if dir_path.exists():
                    dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                    split_idx = int(len(dataset_videos) * self.split_ratio)

                    if self.training:
                        non_violent_videos.extend(dataset_videos[:split_idx])
                    else:
                        non_violent_videos.extend(dataset_videos[split_idx:])
        else:
            for violence_path in self.violence_paths:
                dataset_videos = sorted([f for f in violence_path.rglob('*') if f.is_file()])
                split_idx = int(len(dataset_videos) * self.split_ratio)

                if self.training:
                    violent_videos.extend(dataset_videos[:split_idx])
                else:
                    violent_videos.extend(dataset_videos[split_idx:])

            for non_violence_path in self.non_violence_paths:
                dataset_videos = sorted([f for f in non_violence_path.rglob('*') if f.is_file()])
                split_idx = int(len(dataset_videos) * self.split_ratio)

                if self.training:
                    non_violent_videos.extend(dataset_videos[:split_idx])
                else:
                    non_violent_videos.extend(dataset_videos[split_idx:])

        videos = violent_videos + non_violent_videos
        labels = [1] * len(violent_videos) + [0] * len(non_violent_videos)

        return videos, labels

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

    def _extract_consecutive_clips(self, frames):
        total_frames = len(frames)

        if total_frames < self.n_frames:
            indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
            return [frames[i] for i in indices]

        clips = []

        if total_frames < self.n_frames * self.num_clips:
            step = max(1, (total_frames - self.n_frames) // (self.num_clips - 1))

            for i in range(self.num_clips):
                start_idx = min(i * step, total_frames - self.n_frames)
                clip_frames = frames[start_idx:start_idx + self.n_frames]
                clips.append(clip_frames)
        else:
            step = (total_frames - self.n_frames) // (self.num_clips - 1)

            for i in range(self.num_clips):
                start_idx = i * step
                clip_frames = frames[start_idx:start_idx + self.n_frames]
                clips.append(clip_frames)

        return clips

    def _preprocess_frame(self, frame, target_size=(112, 112)):
        frame = frame.astype(np.float32) / 255.0
        frame = cv2.resize(frame, target_size)
        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        clips = self._extract_consecutive_clips(frames)

        processed_clips = []

        for clip in clips:
            if len(clip) != self.n_frames:
                continue

            processed_frames = [self._preprocess_frame(frame) for frame in clip]
            sequence = np.stack(processed_frames, axis=0)
            sequence = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
            sequence = (sequence - self.mean) / self.std

            processed_clips.append(sequence)

        if len(processed_clips) == 0:
            processed_clips = [torch.zeros(3, self.n_frames, 112, 112)]

        return torch.stack(processed_clips), torch.LongTensor([label])[0]


class HeatmapGenerator3D:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = R3D18Violence(num_classes=2, pretrained=False).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_heatmap_for_sequence(self, sequence, target_class=None):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)

        input_tensor = sequence.unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor, return_cam=True)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        cam_2d = self.model.get_spatial_cam(target_class)

        if cam_2d is None:
            return None, target_class, output.softmax(dim=1)[0].cpu().detach().numpy()

        heatmap_2d = cam_2d[0].cpu().numpy()

        return heatmap_2d, target_class, output.softmax(dim=1)[0].cpu().detach().numpy()

    def visualize_heatmap_on_sequence(self, frames, heatmap, alpha=0.4):
        overlays = []

        mean = torch.tensor(self.config.KINETICS_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.KINETICS_STD).view(3, 1, 1)

        for i in range(frames.size(1)):
            frame = frames[:, i, :, :]

            frame = frame * std + mean
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

            heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays

    def save_visualization(self, output_dir, num_samples=5):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.DATASET_NAME == 'Mix':
            violence_paths, non_violence_paths = self.config.get_mix_paths()
        else:
            violence_paths = self.config.VIOLENCE_PATH
            non_violence_paths = self.config.NON_VIOLENCE_PATH

        dataset = VideoSequenceDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD
        )

        num_samples = min(num_samples, len(dataset))
        indices = range(num_samples)

        count = 0
        for idx in indices:
            try:
                sequence, label = dataset[idx]
                heatmap, pred_class, probs = self.generate_heatmap_for_sequence(sequence)

                if heatmap is None:
                    continue

                overlays = self.visualize_heatmap_on_sequence(sequence, heatmap)

                fig, axes = plt.subplots(4, 4, figsize=(16, 16))
                axes = axes.flatten()

                for i, overlay in enumerate(overlays):
                    if i < len(axes):
                        axes[i].imshow(overlay)
                        axes[i].axis('off')
                        axes[i].set_title(f"Frame {i + 1}")

                for i in range(len(overlays), len(axes)):
                    axes[i].axis('off')

                label_text = "Violence" if pred_class == 1 else "Non-Violence"
                true_label = "Violence" if label == 1 else "Non-Violence"
                confidence = probs[pred_class] * 100

                plt.suptitle(f"Pred: {label_text} ({confidence:.1f}%) | True: {true_label}", fontsize=16)
                plt.tight_layout()

                output_path = output_dir / f"sequence_{idx}_heatmap.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                count += 1
            except Exception as e:
                continue


def evaluate_model_multiview(model_path, config, num_clips=10):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        return 0, [], [], []

    model.eval()

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

    if violence_paths is None or non_violence_paths is None:
        return 0, [], [], []

    val_dataset = MultiViewVideoDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        num_clips=num_clips,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD
    )

    if len(val_dataset) == 0:
        return 0, [], [], []

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for clips, label in tqdm(val_dataset, desc="Evaluating"):
            clips = clips.to(device)
            label = label.to(device)

            clip_outputs = []
            for clip in clips:
                clip = clip.unsqueeze(0)
                output = model(clip)
                clip_outputs.append(output)

            if len(clip_outputs) == 0:
                continue

            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)

            probs = torch.softmax(max_output, dim=1)
            _, predicted = torch.max(max_output, 1)

            total += 1
            correct += (predicted == label).sum().item()

            all_preds.append(predicted.cpu().numpy()[0])
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs[0, 1].cpu().numpy())

    if total == 0:
        return 0, [], [], []

    accuracy = 100 * correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence']))

    return accuracy, all_preds, all_labels, all_probs


def evaluate_model_multiview_with_json(model_path, config, num_clips=10):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        return 0, [], [], [], []

    model.eval()

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

    val_dataset = MultiViewVideoDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        num_clips=num_clips,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD
    )

    if len(val_dataset) == 0:
        return 0, [], [], [], []

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    json_predictions = []

    with torch.no_grad():
        for idx, (clips, label) in enumerate(tqdm(val_dataset, desc="Evaluating")):
            video_path = val_dataset.video_paths[idx]
            video_name = video_path.stem

            clips = clips.to(device)
            label = label.to(device)

            clip_outputs = []
            clip_predictions = []

            for clip_idx, clip in enumerate(clips):
                clip = clip.unsqueeze(0)
                output = model(clip)
                clip_outputs.append(output)

                clip_probs = torch.softmax(output, dim=1)[0].cpu().numpy()

                clip_predictions.append({
                    "algorithmId": "r3d18_violence_detection",
                    "predictions": {
                        "type": "identification",
                        "metadata": {
                            "video_name": video_name,
                            "clip_number": clip_idx,
                            "timestamp": clip_idx * config.N_FRAMES / 30.0,
                            "bbox": [0.0, 0.0, 0.0, 0.0]
                        },
                        "class": ["Non-Violent", "Violent"],
                        "score": [float(clip_probs[0]), float(clip_probs[1])]
                    }
                })

            if len(clip_outputs) == 0:
                continue

            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)

            probs = torch.softmax(max_output, dim=1)
            _, predicted = torch.max(max_output, 1)

            total += 1
            correct += (predicted == label).sum().item()

            all_preds.append(predicted.cpu().numpy()[0])
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs[0, 1].cpu().numpy())

            json_predictions.extend(clip_predictions)

            max_probs = probs[0].cpu().numpy()
            json_predictions.append({
                "algorithmId": "r3d18_violence_detection",
                "predictions": {
                    "type": "identification",
                    "metadata": {
                        "video_name": video_name,
                        "clip_number": "max",
                        "timestamp": 0.0,
                        "bbox": [0.0, 0.0, 0.0, 0.0]
                    },
                    "class": ["Non-Violent", "Violent"],
                    "score": [float(max_probs[0]), float(max_probs[1])]
                }
            })

    if total == 0:
        return 0, [], [], [], []

    accuracy = 100 * correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence']))

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True, parents=True)

    json_path = results_dir / f"results_{config.DATASET_NAME.lower()}_multiview.json"

    with open(json_path, 'w') as f:
        json.dump(json_predictions, f, indent=2)

    return accuracy, all_preds, all_labels, all_probs, json_predictions


def main():
    config = R3DTransferConfig(dataset_name="Crowd")
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        return

    accuracy, preds, labels, probs = evaluate_model_multiview(model_path, config, num_clips=10)

    generator = HeatmapGenerator3D(model_path, config)
    output_dir = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=5)


if __name__ == "__main__":
    main()