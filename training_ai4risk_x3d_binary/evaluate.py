import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import json

from model import X3DViolence
from dataset import X3DVideoDataset
from config import X3DConfig


class MultiViewX3DDataset:
    def __init__(self, violence_path, non_violence_path,
                 num_frames=16, temporal_stride=3,
                 split_ratio=0.8, training=False, num_clips=10,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42, use_crop=False):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
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
        else:
            raise ValueError("Only AI4RiSK multiclass dataset is supported")

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        violent_videos = []
        non_violent_videos = []

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
        temporal_window = self.num_frames * self.temporal_stride

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
            frame_indices = clip_indices[::self.temporal_stride][:self.num_frames]

            if len(frame_indices) < self.num_frames:
                padding_needed = self.num_frames - len(frame_indices)
                for i in range(padding_needed):
                    frame_indices.append(frame_indices[i % len(frame_indices)])

            selected_frames = [frames[i] for i in frame_indices]
            processed_frames = [self._preprocess_frame(frame) for frame in selected_frames]
            sequence = np.stack(processed_frames, axis=0)
            tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std
            processed_clips.append(tensor)

        if len(processed_clips) == 0:
            processed_clips = [torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)]

        label = torch.LongTensor([label])[0]
        return processed_clips, label


class HeatmapGenerator3DX3D:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = X3DViolence(
            num_classes=2,
            pretrained=False,
            x3d_version=config.X3D_VERSION
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_heatmap_for_sequence(self, frames):
        frames = frames.unsqueeze(0).to(self.device)
        frames.requires_grad = True

        outputs = self.model(frames, return_cam=True)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        target_output = outputs[0, pred_class]
        target_output.backward()

        cam = self.model.get_spatial_cam(pred_class)
        if cam is not None:
            cam = cam[0].cpu().numpy()

        return cam, pred_class, probs[0].detach().cpu().numpy()

    def visualize_heatmap_on_sequence(self, frames, heatmap, alpha=0.5):
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

        dataset = X3DVideoDataset(
            violence_path=self.config.VIOLENCE_PATH,
            non_violence_path=self.config.NON_VIOLENCE_PATH,
            num_frames=self.config.NUM_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP
        )

        num_samples = min(num_samples, len(dataset))
        count_success = 0
        count_fail = 0

        for idx in range(num_samples):
            try:
                sequence, label = dataset[idx]
                heatmap, pred_class, probs = self.generate_heatmap_for_sequence(sequence)

                if heatmap is None:
                    count_fail += 1
                    continue

                overlays = self.visualize_heatmap_on_sequence(sequence, heatmap)

                num_frames_to_show = min(8, len(overlays))
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes = axes.flatten()

                for i in range(num_frames_to_show):
                    axes[i].imshow(overlays[i])
                    axes[i].axis('off')
                    axes[i].set_title(f"Frame {i + 1}")

                for i in range(num_frames_to_show, len(axes)):
                    axes[i].axis('off')

                label_text = "Violence" if pred_class == 1 else "Non-Violence"
                true_label = "Violence" if label.item() == 1 else "Non-Violence"
                confidence = probs[pred_class] * 100

                plt.suptitle(f"Pred: {label_text} ({confidence:.1f}%) | True: {true_label}", fontsize=16)
                plt.tight_layout()

                output_path = output_dir / f"sequence_{idx}_label{label.item()}_pred{pred_class}.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                count_success += 1

            except Exception as e:
                print(f"ERROR sample {idx}: {e}")
                count_fail += 1

        print(f"Visualization complete: {count_success} success, {count_fail} failed")
        print(f"Output: {output_dir}")


def evaluate_model_multiview(model_path, config, num_clips=10):
    print(f"\n{'=' * 60}")
    print(f"EVALUATING X3D MODEL - BINARY METRICS")
    print(f"{'=' * 60}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = X3DViolence(
        num_classes=2,
        pretrained=False,
        x3d_version=config.X3D_VERSION
    ).to(device)

    checkpoint_epoch = None
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_epoch = checkpoint.get('epoch', None)
        print(f"Model loaded. Epoch: {checkpoint_epoch}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], [], []

    model.eval()

    if config.VIOLENCE_PATH is None or config.NON_VIOLENCE_PATH is None:
        print("ERROR: paths not configured")
        return 0, [], [], []

    try:
        val_dataset = MultiViewX3DDataset(
            violence_path=config.VIOLENCE_PATH,
            non_violence_path=config.NON_VIOLENCE_PATH,
            num_frames=config.NUM_FRAMES,
            temporal_stride=config.TEMPORAL_STRIDE,
            split_ratio=config.SPLIT_RATIO,
            training=False,
            num_clips=num_clips,
            mean=config.KINETICS_MEAN,
            std=config.KINETICS_STD,
            crop_size=config.CROP_SIZE,
            use_crop=config.USE_CROP
        )
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], [], []

    print(f"Total validation videos: {len(val_dataset)}")

    if len(val_dataset) == 0:
        print("ERROR: No videos in validation set!")
        return 0, [], [], []

    all_preds = []
    all_labels = []
    all_violence_probs = []

    with torch.no_grad():
        for clips, label in tqdm(val_dataset, desc="Evaluating"):
            binary_label = label.item()

            clip_outputs = []
            for clip in clips:
                clip_input = clip.unsqueeze(0).to(device)
                output = model(clip_input)
                clip_outputs.append(output)

            if len(clip_outputs) == 0:
                continue

            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)
            probs = torch.softmax(max_output, dim=1)
            predicted = torch.argmax(max_output, dim=1).item()

            all_preds.append(predicted)
            all_labels.append(binary_label)
            all_violence_probs.append(probs[0, 1].cpu().item())

    if len(all_labels) == 0:
        print("ERROR: No videos processed!")
        return 0, [], [], []

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

    try:
        sns.set(font_scale=1.5)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Violence', 'Violence'], 
                    yticklabels=['Non-Violence', 'Violence'],
                    annot_kws={"size": 35})
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title('Confusion Matrix', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cm_path = Path(config.SAVE_DIR) / "confusion_matrix.jpg"
        plt.savefig(cm_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        sns.reset_orig()
        print(f"Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Confusion matrix save error: {e}")

    roc_auc = None
    try:
        roc_auc = roc_auc_score(all_labels, all_violence_probs) * 100
        
        fpr, tpr, _ = roc_curve(all_labels, all_violence_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc/100:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_curve_path = Path(config.SAVE_DIR) / "roc_curve.jpg"
        plt.savefig(roc_curve_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {roc_curve_path}")
        
    except Exception as e:
        print(f"ROC-AUC error: {e}")

    print(f"\n{'=' * 60}")
    print(f"BINARY EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total videos:  {len(all_labels)}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"Precision:     {precision:.2f}%")
    print(f"Recall:        {recall:.2f}%")
    print(f"F1-Score:      {f1:.2f}%")
    print(f"Specificity:   {specificity:.2f}%")
    print(f"NPV:           {npv:.2f}%")
    if roc_auc is not None:
        print(f"ROC-AUC:       {roc_auc:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-V  Violence")
    print(f"Actual Non-V     {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"       Violence  {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    print(f"\nTP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence'], zero_division=0)}")

    results = {
        "model_path": str(model_path),
        "checkpoint_epoch": checkpoint_epoch,
        "num_clips_per_video": num_clips,
        "total_videos": len(all_labels),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "negative_predictive_value": round(npv, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

    results_path = Path(config.SAVE_DIR) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_path}")

    return accuracy, all_preds, all_labels, all_violence_probs


def main():
    config = X3DConfig()
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path.absolute()}")
        return

    print(f"Model: {model_path}")
    print(f"Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

    accuracy, preds, labels, probs = evaluate_model_multiview(model_path, config, num_clips=4)

    generator = HeatmapGenerator3DX3D(model_path, config)
    output_dir = Path(f"heatmap_visualizations_x3d_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=20)


if __name__ == "__main__":
    main()