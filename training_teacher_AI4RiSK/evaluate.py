import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

from model import SlowFastViolence
from dataset import SlowFastVideoDataset
from config import SlowFastConfig


class MultiViewSlowFastDataset:
    def __init__(self, violence_path, non_violence_path,
                 num_frames_slow=8, num_frames_fast=32,
                 alpha=4, tau_slow=16, tau_fast=2,
                 split_ratio=0.75, training=False, num_clips=10,
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
        self.num_clips = num_clips
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
        temporal_window = self.num_frames_fast * self.tau_fast

        if total_frames < temporal_window:
            indices = [i % total_frames for i in range(temporal_window)]
            return [indices]

        clips = []

        if total_frames < temporal_window * self.num_clips:
            step = max(1, (total_frames - temporal_window) // (self.num_clips - 1))

            for i in range(self.num_clips):
                start_idx = min(i * step, total_frames - temporal_window)
                clip_indices = list(range(start_idx, start_idx + temporal_window))
                clips.append(clip_indices)
        else:
            step = (total_frames - temporal_window) // (self.num_clips - 1)

            for i in range(self.num_clips):
                start_idx = i * step
                clip_indices = list(range(start_idx, start_idx + temporal_window))
                clips.append(clip_indices)

        return clips

    def _preprocess_frame(self, frame, target_size=256):
        frame = frame.astype(np.float32) / 255.0

        if self.use_crop:
            h, w = frame.shape[:2]
            scale = target_size / min(h, w)
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
        clip_indices_list = self._extract_consecutive_clips(frames)

        processed_clips = []

        for clip_indices in clip_indices_list:
            fast_indices = clip_indices[::self.tau_fast][:self.num_frames_fast]
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

            processed_slow = [self._preprocess_frame(frame) for frame in slow_frames]
            processed_fast = [self._preprocess_frame(frame) for frame in fast_frames]

            slow_sequence = np.stack(processed_slow, axis=0)
            fast_sequence = np.stack(processed_fast, axis=0)

            slow_tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
            fast_tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)

            slow_tensor = (slow_tensor - self.mean) / self.std
            fast_tensor = (fast_tensor - self.mean) / self.std

            processed_clips.append([slow_tensor, fast_tensor])

        if len(processed_clips) == 0:
            processed_clips = [[
                torch.zeros(3, self.num_frames_slow, self.crop_size, self.crop_size),
                torch.zeros(3, self.num_frames_fast, self.crop_size, self.crop_size)
            ]]

        return processed_clips, torch.LongTensor([label])[0]


class HeatmapGenerator3DSlowFast:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = SlowFastViolence(
            num_classes=2,
            pretrained=False,
            alpha=config.SLOWFAST_ALPHA
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_heatmap_for_sequence(self, sequence, target_class=None):
        if not isinstance(sequence, list):
            raise ValueError("Sequence must be [slow_pathway, fast_pathway]")

        slow_pathway = sequence[0].unsqueeze(0).to(self.device)
        fast_pathway = sequence[1].unsqueeze(0).to(self.device)

        slow_pathway.requires_grad = True
        fast_pathway.requires_grad = True

        output = self.model([slow_pathway, fast_pathway], return_cam=True)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        cam_2d = self.model.get_spatial_cam(target_class, pathway='slow')

        if cam_2d is None:
            return None, target_class, output.softmax(dim=1)[0].cpu().detach().numpy()

        heatmap_2d = cam_2d[0].cpu().numpy()

        return heatmap_2d, target_class, output.softmax(dim=1)[0].cpu().detach().numpy()

    def visualize_heatmap_on_sequence(self, frames, heatmap, alpha=0.4):
        overlays = []

        mean = torch.tensor(self.config.KINETICS_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.KINETICS_STD).view(3, 1, 1)

        slow_frames = frames[0]

        for i in range(slow_frames.size(1)):
            frame = slow_frames[:, i, :, :]

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

        dataset = SlowFastVideoDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            num_frames_slow=self.config.NUM_FRAMES_SLOW,
            num_frames_fast=self.config.NUM_FRAMES_FAST,
            alpha=self.config.SLOWFAST_ALPHA,
            tau_slow=self.config.SLOWFAST_TAU_SLOW,
            tau_fast=self.config.SLOWFAST_TAU_FAST,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP
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

                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes = axes.flatten()

                for i, overlay in enumerate(overlays):
                    if i < len(axes):
                        axes[i].imshow(overlay)
                        axes[i].axis('off')
                        axes[i].set_title(f"Slow Frame {i + 1}")

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

    model = SlowFastViolence(
        num_classes=2,
        pretrained=False,
        alpha=config.SLOWFAST_ALPHA
    ).to(device)

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

    val_dataset = MultiViewSlowFastDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        num_frames_slow=config.NUM_FRAMES_SLOW,
        num_frames_fast=config.NUM_FRAMES_FAST,
        alpha=config.SLOWFAST_ALPHA,
        tau_slow=config.SLOWFAST_TAU_SLOW,
        tau_fast=config.SLOWFAST_TAU_FAST,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        num_clips=num_clips,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD,
        crop_size=config.CROP_SIZE,
        use_crop=config.USE_CROP
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
            label = label.to(device)

            clip_outputs = []
            for clip in clips:
                slow_pathway = clip[0].unsqueeze(0).to(device)
                fast_pathway = clip[1].unsqueeze(0).to(device)

                output = model([slow_pathway, fast_pathway])
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


def main():
    config = SlowFastConfig(dataset_name="Crowd")
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        return

    accuracy, preds, labels, probs = evaluate_model_multiview(model_path, config, num_clips=10)

    generator = HeatmapGenerator3DSlowFast(model_path, config)
    output_dir = Path(f"heatmap_visualizations_slowfast_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=5)


if __name__ == "__main__":
    main()