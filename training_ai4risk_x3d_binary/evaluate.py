from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from config import X3DConfig
from dataset import FrameArray, X3DVideoDataset
from model import X3DViolence


logger: logging.Logger = logging.getLogger(__name__)

EvaluationReturn = tuple[float, list[int], list[int], list[float]]
ClipDatasetItem = tuple[list[torch.Tensor], torch.Tensor]


class MultiViewX3DDataset:
    """Builds multiple temporal clips from each validation video for multi-view evaluation."""

    def __init__(
        self,
        violence_path: dict[str, Path | list[str] | str],
        non_violence_path: Optional[dict[str, Path | list[str] | str]],
        num_frames: int = 16,
        temporal_stride: int = 3,
        split_ratio: float = 0.8,
        training: bool = False,
        num_clips: int = 10,
        mean: Sequence[float] = (0.45, 0.45, 0.45),
        std: Sequence[float] = (0.225, 0.225, 0.225),
        crop_size: int = 224,
        seed: int = 42,
        use_crop: bool = False,
    ) -> None:
        """Stores clip extraction parameters and builds the validation video list."""

        self.num_frames: int = num_frames
        self.temporal_stride: int = temporal_stride
        self.split_ratio: float = split_ratio
        self.training: bool = training
        self.num_clips: int = num_clips
        self.crop_size: int = crop_size
        self.seed: int = seed
        self.use_crop: bool = use_crop
        self.mean: torch.Tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std: torch.Tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get("type") == "multiclass":
            self.dataset_type: str = "multiclass"
            self.base_path: Path = Path(violence_path["path"]) # type: ignore
            self.violence_dirs: list[str] = list(violence_path["violence_dirs"]) # type: ignore
            self.non_violence_dirs: list[str] = list(violence_path["non_violence_dirs"]) # type: ignore
        else:
            raise ValueError("Only the AI4RiSK multiclass dataset format is supported.")

        self.video_paths: list[Path]
        self.labels: list[int]
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self) -> tuple[list[Path], list[int]]:
        """Collects validation or training video paths and binary labels."""

        violent_videos: list[Path] = []
        non_violent_videos: list[Path] = []
        base_path: Path = Path(self.base_path)

        for dir_name in self.violence_dirs:
            dir_path: Path = base_path / dir_name
            if dir_path.exists():
                dataset_videos: list[Path] = sorted(file_path for file_path in dir_path.rglob("*") if file_path.is_file())
                split_idx: int = int(len(dataset_videos) * self.split_ratio)
                selected_videos: list[Path] = dataset_videos[:split_idx] if self.training else dataset_videos[split_idx:]
                violent_videos.extend(selected_videos)

        for dir_name in self.non_violence_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted(file_path for file_path in dir_path.rglob("*") if file_path.is_file())
                split_idx = int(len(dataset_videos) * self.split_ratio)
                selected_videos = dataset_videos[:split_idx] if self.training else dataset_videos[split_idx:]
                non_violent_videos.extend(selected_videos)

        videos: list[Path] = violent_videos + non_violent_videos
        labels: list[int] = [1] * len(violent_videos) + [0] * len(non_violent_videos)
        return videos, labels

    def _extract_frames(self, video_path: Path) -> list[FrameArray]:
        """Reads a video file and returns RGB frames."""

        capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        frames: list[FrameArray] = []

        while True:
            success: bool
            frame: Optional[FrameArray]
            success, frame = capture.read()
            if not success or frame is None:
                break

            rgb_frame: FrameArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        capture.release()
        return frames

    def _extract_consecutive_clips(self, frames: list[FrameArray]) -> list[list[int]]:
        """Returns frame index windows for evenly spaced temporal clips."""

        total_frames: int = len(frames)
        temporal_window: int = self.num_frames * self.temporal_stride

        if total_frames == 0:
            return []

        if total_frames < temporal_window:
            padded_indices: list[int] = list(range(total_frames))
            last_index: int = total_frames - 1
            while len(padded_indices) < temporal_window:
                padded_indices.append(last_index)
            return [padded_indices]

        clips: list[list[int]] = []

        if self.num_clips <= 1:
            center_start: int = max(0, (total_frames - temporal_window) // 2)
            return [list(range(center_start, center_start + temporal_window))]

        if total_frames < temporal_window * self.num_clips:
            step: int = max(1, (total_frames - temporal_window) // (self.num_clips - 1))
        else:
            step = (total_frames - temporal_window) // (self.num_clips - 1)

        for clip_index in range(self.num_clips):
            start_idx: int = min(clip_index * step, total_frames - temporal_window)
            clip_indices: list[int] = list(range(start_idx, start_idx + temporal_window))
            clips.append(clip_indices)

        return clips

    def _preprocess_frame(self, frame: FrameArray, target_size: int = 256) -> FrameArray:
        """Scales and center-crops or resizes a frame for X3D input."""

        processed_frame: FrameArray = frame.astype(np.float32) / 255.0

        if self.use_crop:
            height: int
            width: int
            height, width = processed_frame.shape[:2]
            scale: float = target_size / min(height, width)
            new_height: int = int(height * scale)
            new_width: int = int(width * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            height, width = processed_frame.shape[:2]
            top: int = (height - self.crop_size) // 2
            left: int = (width - self.crop_size) // 2
            processed_frame = processed_frame[top : top + self.crop_size, left : left + self.crop_size]
        else:
            processed_frame = cv2.resize(processed_frame, (self.crop_size, self.crop_size))

        return processed_frame

    def __len__(self) -> int:
        """Returns the number of videos in the selected split."""

        return len(self.video_paths)

    def __getitem__(self, idx: int) -> ClipDatasetItem:
        """Returns all processed clips for one video and its label."""

        video_path: Path = self.video_paths[idx]
        label_value: int = self.labels[idx]
        frames: list[FrameArray] = self._extract_frames(video_path)
        clip_indices_list: list[list[int]] = self._extract_consecutive_clips(frames)
        processed_clips: list[torch.Tensor] = []

        for clip_indices in clip_indices_list:
            frame_indices: list[int] = clip_indices[:: self.temporal_stride][: self.num_frames]

            if len(frame_indices) < self.num_frames and frame_indices:
                padding_needed: int = self.num_frames - len(frame_indices)
                for padding_index in range(padding_needed):
                    frame_indices.append(frame_indices[padding_index % len(frame_indices)])

            if not frame_indices:
                continue

            selected_frames: list[FrameArray] = [frames[index] for index in frame_indices]
            processed_frames: list[FrameArray] = [self._preprocess_frame(frame) for frame in selected_frames]
            sequence: FrameArray = np.stack(processed_frames, axis=0)
            tensor: torch.Tensor = torch.as_tensor(sequence, dtype=torch.float32).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std
            processed_clips.append(tensor)

        if len(processed_clips) == 0:
            empty_clip: torch.Tensor = torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)
            processed_clips = [empty_clip]

        label: torch.Tensor = torch.tensor(label_value, dtype=torch.long)
        return processed_clips, label


class HeatmapGenerator3DX3D:
    """Generates Grad-CAM heatmaps and overlay images for X3D predictions."""

    def __init__(self, model_path: Path, config: X3DConfig) -> None:
        """Loads a trained X3D checkpoint for Grad-CAM visualization."""

        self.config: X3DConfig = config
        self.device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model: X3DViolence = X3DViolence(
            num_classes=2,
            pretrained=False,
            x3d_version=config.X3D_VERSION,
        ).to(self.device)

        checkpoint: dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate_heatmap_for_sequence(self, frames: torch.Tensor) -> tuple[Optional[FrameArray], int, FrameArray]:
        """Computes a spatial Grad-CAM heatmap for a single normalized video tensor."""

        model_input: torch.Tensor = frames.unsqueeze(0).to(self.device)
        model_input.requires_grad = True

        outputs: torch.Tensor = self.model(model_input, return_cam=True)
        probabilities: torch.Tensor = torch.softmax(outputs, dim=1)
        predicted_class: int = int(torch.argmax(probabilities, dim=1).item())
        target_output: torch.Tensor = outputs[0, predicted_class]
        target_output.backward()

        cam: Optional[torch.Tensor] = self.model.get_spatial_cam(predicted_class)
        heatmap: Optional[FrameArray] = cam[0].detach().cpu().numpy() if cam is not None else None
        probability_values: FrameArray = probabilities[0].detach().cpu().numpy()

        return heatmap, predicted_class, probability_values

    def visualize_heatmap_on_sequence(
        self,
        frames: torch.Tensor,
        heatmap: FrameArray,
        alpha: float = 0.5,
    ) -> list[FrameArray]:
        """Overlays a heatmap on each frame of a normalized video tensor."""

        overlays: list[FrameArray] = []
        mean: torch.Tensor = torch.tensor(self.config.KINETICS_MEAN, dtype=torch.float32).view(3, 1, 1)
        std: torch.Tensor = torch.tensor(self.config.KINETICS_STD, dtype=torch.float32).view(3, 1, 1)

        for frame_index in range(frames.size(1)):
            frame: torch.Tensor = frames[:, frame_index, :, :]
            frame = frame * std + mean
            frame_array: FrameArray = frame.permute(1, 2, 0).cpu().numpy()
            frame_array = np.clip(frame_array * 255, 0, 255).astype(np.uint8)

            heatmap_resized: FrameArray = cv2.resize(heatmap, (frame_array.shape[1], frame_array.shape[0]))
            heatmap_colored: FrameArray = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay: FrameArray = cv2.addWeighted(frame_array, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays

    def save_visualization(self, output_dir: Path, num_samples: int = 5, verbose: bool = True) -> None:
        """Saves Grad-CAM visualization grids for a sample of validation videos."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset: X3DVideoDataset = X3DVideoDataset(
            violence_path=self.config.VIOLENCE_PATH, # type: ignore
            non_violence_path=self.config.NON_VIOLENCE_PATH,
            num_frames=self.config.NUM_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP,
        )

        sample_count: int = min(num_samples, len(dataset))
        success_count: int = 0
        failure_count: int = 0

        for sample_index in range(sample_count):
            try:
                sequence: torch.Tensor
                label: torch.Tensor
                sequence, label = dataset[sample_index]
                heatmap: Optional[FrameArray]
                predicted_class: int
                probabilities: FrameArray
                heatmap, predicted_class, probabilities = self.generate_heatmap_for_sequence(sequence)

                if heatmap is None:
                    failure_count += 1
                    continue

                overlays: list[FrameArray] = self.visualize_heatmap_on_sequence(sequence, heatmap)
                frames_to_show: int = min(8, len(overlays))
                figure: plt.Figure
                axes_array: np.ndarray
                figure, axes_array = plt.subplots(2, 4, figsize=(16, 8))
                axes: np.ndarray = axes_array.flatten()

                for frame_index in range(frames_to_show):
                    axes[frame_index].imshow(overlays[frame_index])
                    axes[frame_index].axis("off")
                    axes[frame_index].set_title(f"Frame {frame_index + 1}")

                for frame_index in range(frames_to_show, len(axes)):
                    axes[frame_index].axis("off")

                predicted_label: str = "Violence" if predicted_class == 1 else "Non-Violence"
                true_label: str = "Violence" if label.item() == 1 else "Non-Violence"
                confidence: float = float(probabilities[predicted_class] * 100)
                output_path: Path = output_dir / f"sequence_{sample_index}_label{label.item()}_pred{predicted_class}.png"

                plt.suptitle(f"Predicted: {predicted_label} ({confidence:.1f}%) | True: {true_label}", fontsize=16)
                plt.tight_layout()
                plt.savefig(output_path, dpi=100, bbox_inches="tight")
                plt.close(figure)
                success_count += 1

            except Exception as exc:
                failure_count += 1
                if verbose:
                    logger.warning("Could not save heatmap for sample %d: %s", sample_index, exc)

        if verbose:
            logger.info("Saved %d heatmap visualizations to %s. Failed samples: %d.", success_count, output_dir, failure_count)


def _save_confusion_matrix(cm: FrameArray, output_path: Path) -> None:
    """Saves a confusion matrix figure to disk."""

    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Violence", "Violence"],
        yticklabels=["Non-Violence", "Violence"],
        annot_kws={"size": 35},
    )
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.title("Confusion Matrix", fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    sns.reset_orig()


def _save_roc_curve(labels: list[int], probabilities: list[float], roc_auc: float, output_path: Path) -> None:
    """Saves a ROC curve figure to disk."""

    false_positive_rate: FrameArray
    true_positive_rate: FrameArray
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC curve (area = {roc_auc / 100:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model_multiview(
    model_path: Path,
    config: X3DConfig,
    num_clips: int = 10,
    verbose: bool = True,
) -> EvaluationReturn:
    """Evaluates a trained model by aggregating predictions over multiple clips per video."""

    device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model: X3DViolence = X3DViolence(
        num_classes=2,
        pretrained=False,
        x3d_version=config.X3D_VERSION,
    ).to(device)

    try:
        checkpoint: dict = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_epoch: Optional[int] = checkpoint.get("epoch")
    except Exception as exc:
        logger.error("Could not load model checkpoint: %s", exc)
        return 0.0, [], [], []

    model.eval()

    if config.VIOLENCE_PATH is None or config.NON_VIOLENCE_PATH is None:
        logger.error("Dataset paths are not configured.")
        return 0.0, [], [], []

    try:
        validation_dataset: MultiViewX3DDataset = MultiViewX3DDataset(
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
            use_crop=config.USE_CROP,
        )
    except Exception as exc:
        logger.error("Could not create validation dataset: %s", exc)
        return 0.0, [], [], []

    if len(validation_dataset) == 0:
        logger.error("The validation split contains no videos.")
        return 0.0, [], [], []

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_violence_probs: list[float] = []

    with torch.no_grad():
        progress_bar: tqdm = tqdm(validation_dataset, desc="Evaluating", disable=not verbose, leave=False)

        for clips, label in progress_bar:
            binary_label: int = int(label.item())
            clip_outputs: list[torch.Tensor] = []

            for clip in clips:
                clip_input: torch.Tensor = clip.unsqueeze(0).to(device)
                output: torch.Tensor = model(clip_input)
                clip_outputs.append(output)

            if len(clip_outputs) == 0:
                continue

            max_output: torch.Tensor
            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)
            probabilities: torch.Tensor = torch.softmax(max_output, dim=1)
            predicted: int = int(torch.argmax(max_output, dim=1).item())

            all_preds.append(predicted)
            all_labels.append(binary_label)
            all_violence_probs.append(float(probabilities[0, 1].cpu().item()))

    if len(all_labels) == 0:
        logger.error("No videos were processed during evaluation.")
        return 0.0, [], [], []

    accuracy: float = float(accuracy_score(all_labels, all_preds) * 100)
    precision: float = float(precision_score(all_labels, all_preds, zero_division=0) * 100)
    recall: float = float(recall_score(all_labels, all_preds, zero_division=0) * 100)
    f1: float = float(f1_score(all_labels, all_preds, zero_division=0) * 100)
    cm: FrameArray = confusion_matrix(all_labels, all_preds)
    tn: int
    fp: int
    fn: int
    tp: int
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    specificity: float = float(tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    negative_predictive_value: float = float(tn / (tn + fn) * 100) if (tn + fn) > 0 else 0.0
    roc_auc: Optional[float] = None

    confusion_matrix_path: Path = Path(config.SAVE_DIR) / "confusion_matrix.jpg"
    try:
        _save_confusion_matrix(cm, confusion_matrix_path)
    except Exception as exc:
        logger.warning("Could not save confusion matrix: %s", exc)

    roc_curve_path: Path = Path(config.SAVE_DIR) / "roc_curve.jpg"
    try:
        roc_auc = float(roc_auc_score(all_labels, all_violence_probs) * 100)
        _save_roc_curve(all_labels, all_violence_probs, roc_auc, roc_curve_path)
    except Exception as exc:
        logger.warning("Could not save ROC curve: %s", exc)

    report: str = classification_report(
        all_labels,
        all_preds,
        target_names=["Non-Violence", "Violence"],
        zero_division=0,
    )

    results: dict = {
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
            "negative_predictive_value": round(negative_predictive_value, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "classification_report": report,
    }

    results_path: Path = Path(config.SAVE_DIR) / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    if verbose:
        logger.info(
            "Evaluation complete | videos=%d | accuracy=%.2f%% | precision=%.2f%% | recall=%.2f%% | f1=%.2f%%",
            len(all_labels),
            accuracy,
            precision,
            recall,
            f1,
        )
        logger.info("Evaluation results saved to %s.", results_path)

    return accuracy, all_preds, all_labels, all_violence_probs


def main() -> None:
    """Runs multi-view evaluation and Grad-CAM visualization with the default configuration."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config: X3DConfig = X3DConfig()
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path: Path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    if not model_path.exists():
        logger.error("Model checkpoint not found at %s.", model_path.absolute())
        return

    evaluate_model_multiview(model_path, config, num_clips=4)

    generator: HeatmapGenerator3DX3D = HeatmapGenerator3DX3D(model_path, config)
    output_dir: Path = Path(f"heatmap_visualizations_x3d_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=20)


if __name__ == "__main__":
    main()