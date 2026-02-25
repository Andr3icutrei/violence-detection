import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    recall_score, precision_score
)
import json

from model import X3DViolence
from dataset import X3DVideoDataset, MultiViewX3DDataset
from config import X3DConfig


class HeatmapGenerator3DX3D:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = X3DViolence(
            num_classes=config.NUM_CLASSES,
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
            dataset_info=self.config.DATASET_INFO,
            num_frames=self.config.NUM_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            split_ratio=self.config.SPLIT_RATIO,
            split_seed=self.config.SPLIT_SEED,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP,
            max_retries=self.config.DATASET_MAX_RETRIES
        )

        num_samples = min(num_samples, len(dataset))

        for idx in range(num_samples):
            try:
                sequence, label = dataset[idx]
                heatmap, pred_class, probs = self.generate_heatmap_for_sequence(sequence)

                if heatmap is None:
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

                binary_pred = 0 if pred_class == 0 else 1
                binary_label = 0 if label.item() == 0 else 1
                pred_name = "Violence" if binary_pred == 1 else "Non-Violence"
                true_name = "Violence" if binary_label == 1 else "Non-Violence"
                confidence = probs[pred_class] * 100

                plt.suptitle(
                    f"Pred: {pred_name} ({confidence:.1f}%) | True: {true_name}",
                    fontsize=16
                )
                plt.tight_layout()

                output_path = output_dir / f"sequence_{idx}_label{binary_label}_pred{binary_pred}.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()

            except Exception:
                continue


def evaluate_model_multiview(model_path, config, num_clips=None):
    print(f"\n{'=' * 60}")
    print(f"EVALUATING MULTICLASS X3D MODEL - BINARY METRICS")
    print(f"{'=' * 60}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if num_clips is None:
        num_clips = config.MULTIVIEW_NUM_CLIPS

    model = X3DViolence(
        num_classes=config.NUM_CLASSES,
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
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], [], []

    model.eval()

    try:
        val_dataset = MultiViewX3DDataset(
            dataset_info=config.DATASET_INFO,
            num_frames=config.NUM_FRAMES,
            temporal_stride=config.TEMPORAL_STRIDE,
            split_ratio=config.SPLIT_RATIO,
            split_seed=config.SPLIT_SEED,
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
        print("ERROR: Validation dataset is empty.")
        return 0, [], [], []

    all_binary_preds = []
    all_binary_labels = []
    all_violence_probs = []

    with torch.no_grad():
        for clips, label in tqdm(val_dataset, desc="Evaluating"):
            raw_label = label.item()
            binary_label = 0 if raw_label == 0 else 1

            clip_outputs = []
            for clip in clips:
                clip_input = clip.unsqueeze(0).to(device)
                output = model(clip_input)
                clip_outputs.append(output)

            if not clip_outputs:
                continue

            stacked = torch.stack(clip_outputs, dim=0)
            avg_output = torch.mean(stacked, dim=0)

            probs = torch.softmax(avg_output, dim=1)
            multiclass_pred = torch.argmax(avg_output, dim=1).item()
            binary_pred = 0 if multiclass_pred == 0 else 1

            violence_prob = probs[0, 1:].sum().cpu().item()

            all_binary_preds.append(binary_pred)
            all_binary_labels.append(binary_label)
            all_violence_probs.append(violence_prob)

    if not all_binary_labels:
        print("ERROR: No videos processed!")
        return 0, [], [], []

    accuracy = accuracy_score(all_binary_labels, all_binary_preds) * 100
    precision = precision_score(all_binary_labels, all_binary_preds, zero_division=0) * 100
    recall = recall_score(all_binary_labels, all_binary_preds, zero_division=0) * 100
    f1 = f1_score(all_binary_labels, all_binary_preds, zero_division=0) * 100
    cm = confusion_matrix(all_binary_labels, all_binary_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

    roc_auc = None
    try:
        roc_auc = roc_auc_score(all_binary_labels, all_violence_probs) * 100
    except Exception as e:
        print(f"ROC-AUC error: {e}")

    print(f"\n{'=' * 60}")
    print(f"BINARY EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total videos:  {len(all_binary_labels)}")
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
    print(f"\n{classification_report(all_binary_labels, all_binary_preds, target_names=['Non-Violence', 'Violence'], zero_division=0)}")

    results = {
        "model_path": str(model_path),
        "checkpoint_epoch": checkpoint_epoch,
        "num_clips_per_video": num_clips,
        "total_videos": len(all_binary_labels),
        "note": "Multiclass model (classes 1-N mapped to Violence, class 0 = Non-Violence)",
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

    return accuracy, all_binary_preds, all_binary_labels, all_violence_probs


def main():
    config = X3DConfig()
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path.absolute()}")
        return

    print(f"Model: {model_path}")
    print(f"Size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

    accuracy, preds, labels, probs = evaluate_model_multiview(model_path, config, num_clips=4)

    generator = HeatmapGenerator3DX3D(model_path, config)
    output_dir = Path(f"heatmap_visualizations_x3d_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=20)


if __name__ == "__main__":
    main()