from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypeAlias

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset

from config import R3DTransferConfig, DatasetPath
from dataset import VideoSequenceDataset
from model import R3D18Violence

FrameArray: TypeAlias = np.ndarray
PredictionJson: TypeAlias = dict[str, str | dict[str, str | dict[str, str | int | float | list[float]] | list[str] | list[float]]]


class MultiViewVideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load validation videos as multiple temporal clips for multi-view evaluation."""

    def __init__(
        self,
        violence_path: DatasetPath,
        non_violence_path: DatasetPath,
        n_frames: int = 16,
        split_ratio: float = 0.75,
        training: bool = False,
        num_clips: int = 10,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        """Initialize video sources, split options, clip count, and normalization tensors."""
        self.n_frames: int = n_frames
        self.split_ratio: float = split_ratio
        self.training: bool = training
        self.num_clips: int = num_clips

        normalization_mean: list[float] = mean or [0.43216, 0.394666, 0.37645]
        normalization_std: list[float] = std or [0.22803, 0.22145, 0.216989]
        self.mean: torch.Tensor = torch.tensor(normalization_mean).view(3, 1, 1, 1)
        self.std: torch.Tensor = torch.tensor(normalization_std).view(3, 1, 1, 1)

        self.dataset_type: str
        self.base_path: str | Path | None = None
        self.violence_dirs: list[str] | None = None
        self.non_violence_dirs: list[str] | None = None
        self.violence_paths: list[Path] | None = None
        self.non_violence_paths: list[Path] | None = None

        if isinstance(violence_path, dict) and violence_path.get("type") == "multiclass":
            self.dataset_type = "multiclass"
            self.base_path = violence_path["path"]
            self.violence_dirs = violence_path["violence_dirs"]
            self.non_violence_dirs = violence_path["non_violence_dirs"]
        elif isinstance(violence_path, (list, tuple)):
            self.dataset_type = "standard"
            self.violence_paths = [Path(path) for path in violence_path]
            self.non_violence_paths = [Path(path) for path in non_violence_path]
        else:
            self.dataset_type = "standard"
            self.violence_paths = [Path(violence_path)] if violence_path else None
            self.non_violence_paths = [Path(non_violence_path)] if non_violence_path else None

        self.video_paths: list[Path]
        self.labels: list[int]
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self) -> tuple[list[Path], list[int]]:
        """Return validation or training video paths with their binary labels."""
        violent_videos: list[Path] = []
        non_violent_videos: list[Path] = []

        if self.dataset_type == "multiclass":
            base_path: Path = Path(self.base_path)
            violence_dirs: list[str] = self.violence_dirs or []
            non_violence_dirs: list[str] = self.non_violence_dirs or []

            for directory_name in violence_dirs:
                directory_path: Path = base_path / directory_name
                if directory_path.exists():
                    dataset_videos: list[Path] = sorted(file_path for file_path in directory_path.rglob("*") if file_path.is_file())
                    random.seed(R3DTransferConfig.SEED)
                    random.shuffle(dataset_videos)
                    split_index: int = int(len(dataset_videos) * self.split_ratio)
                    violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])

            for directory_name in non_violence_dirs:
                directory_path = base_path / directory_name
                if directory_path.exists():
                    dataset_videos = sorted(file_path for file_path in directory_path.rglob("*") if file_path.is_file())
                    random.seed(R3DTransferConfig.SEED)
                    random.shuffle(dataset_videos)
                    split_index = int(len(dataset_videos) * self.split_ratio)
                    non_violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])
        else:
            violence_paths: list[Path] = self.violence_paths or []
            non_violence_paths: list[Path] = self.non_violence_paths or []

            for violence_path in violence_paths:
                dataset_videos = sorted(file_path for file_path in violence_path.rglob("*") if file_path.is_file())
                random.seed(R3DTransferConfig.SEED)
                random.shuffle(dataset_videos)
                split_index = int(len(dataset_videos) * self.split_ratio)
                violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])

            for non_violence_path in non_violence_paths:
                dataset_videos = sorted(file_path for file_path in non_violence_path.rglob("*") if file_path.is_file())
                random.seed(R3DTransferConfig.SEED)
                random.shuffle(dataset_videos)
                split_index = int(len(dataset_videos) * self.split_ratio)
                non_violent_videos.extend(dataset_videos[:split_index] if self.training else dataset_videos[split_index:])

        videos: list[Path] = violent_videos + non_violent_videos
        labels: list[int] = [1] * len(violent_videos) + [0] * len(non_violent_videos)
        return videos, labels

    def _extract_frames(self, video_path: Path) -> list[FrameArray]:
        """Decode all RGB frames from a video file."""
        video_capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        frames: list[FrameArray] = []

        while True:
            success: bool
            frame: FrameArray
            success, frame = video_capture.read()
            if not success:
                break
            rgb_frame: FrameArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        video_capture.release()
        return frames

    def _extract_consecutive_clips(self, frames: list[FrameArray]) -> list[list[FrameArray]]:
        """Split decoded frames into evenly spaced fixed-length clips."""
        total_frames: int = len(frames)
        clips: list[list[FrameArray]] = []

        if total_frames == 0:
            return clips

        if total_frames < self.n_frames:
            indices: np.ndarray = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
            clips.append([frames[index] for index in indices])
            return clips

        if total_frames < self.n_frames * self.num_clips:
            step: int = max(1, (total_frames - self.n_frames) // max(1, self.num_clips - 1))
        else:
            step = (total_frames - self.n_frames) // max(1, self.num_clips - 1)

        for clip_index in range(self.num_clips):
            start_index: int = min(clip_index * step, total_frames - self.n_frames)
            clip_frames: list[FrameArray] = frames[start_index:start_index + self.n_frames]
            clips.append(clip_frames)

        return clips

    def _preprocess_frame(self, frame: FrameArray, target_size: tuple[int, int] = (112, 112)) -> FrameArray:
        """Scale one RGB frame to 0-1 values and resize it for R3D-18."""
        scaled_frame: FrameArray = frame.astype(np.float32) / 255.0
        resized_frame: FrameArray = cv2.resize(scaled_frame, target_size)
        return resized_frame

    def __len__(self) -> int:
        """Return the number of videos in the evaluation split."""
        return len(self.video_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all sampled clips for one video and the corresponding label."""
        video_path: Path = self.video_paths[index]
        label_value: int = self.labels[index]
        frames: list[FrameArray] = self._extract_frames(video_path)
        clips: list[list[FrameArray]] = self._extract_consecutive_clips(frames)
        processed_clips: list[torch.Tensor] = []

        for clip in clips:
            if len(clip) != self.n_frames:
                continue

            processed_frames: list[FrameArray] = [self._preprocess_frame(frame) for frame in clip]
            sequence_array: FrameArray = np.stack(processed_frames, axis=0)
            sequence_tensor: torch.Tensor = torch.FloatTensor(sequence_array).permute(3, 0, 1, 2)
            sequence_tensor = (sequence_tensor - self.mean) / self.std
            processed_clips.append(sequence_tensor)

        if not processed_clips:
            processed_clips = [torch.zeros(3, self.n_frames, 112, 112)]

        label_tensor: torch.Tensor = torch.LongTensor([label_value])[0]
        return torch.stack(processed_clips), label_tensor


class HeatmapGenerator3D:
    """Generate and save spatial Grad-CAM visualizations for R3D-18 video clips."""

    def __init__(self, model_path: str | Path, config: R3DTransferConfig) -> None:
        """Load a trained model checkpoint for heatmap generation."""
        self.config: R3DTransferConfig = config
        self.device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model: R3D18Violence = R3D18Violence(num_classes=2, pretrained=False).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate_heatmap_for_sequence(
        self,
        sequence: torch.Tensor | np.ndarray,
        target_class: int | None = None,
    ) -> tuple[np.ndarray | None, int, np.ndarray]:
        """Create a 2D Grad-CAM heatmap and prediction probabilities for one sequence."""
        sequence_tensor: torch.Tensor = sequence if isinstance(sequence, torch.Tensor) else torch.FloatTensor(sequence)
        input_tensor: torch.Tensor = sequence_tensor.unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        output: torch.Tensor = self.model(input_tensor, return_cam=True)
        selected_class: int = int(output.argmax(dim=1).item()) if target_class is None else target_class

        self.model.zero_grad()
        output[0, selected_class].backward()

        cam_2d: torch.Tensor | None = self.model.get_spatial_cam(selected_class)
        probabilities: np.ndarray = output.softmax(dim=1)[0].cpu().detach().numpy()

        if cam_2d is None:
            return None, selected_class, probabilities

        heatmap_2d: np.ndarray = cam_2d[0].cpu().numpy()
        return heatmap_2d, selected_class, probabilities

    def visualize_heatmap_on_sequence(
        self,
        frames: torch.Tensor,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> list[FrameArray]:
        """Overlay a Grad-CAM heatmap on every frame of a normalized sequence."""
        overlays: list[FrameArray] = []
        mean: torch.Tensor = torch.tensor(self.config.KINETICS_MEAN).view(3, 1, 1)
        std: torch.Tensor = torch.tensor(self.config.KINETICS_STD).view(3, 1, 1)

        for frame_index in range(frames.size(1)):
            frame_tensor: torch.Tensor = frames[:, frame_index, :, :]
            frame_tensor = frame_tensor * std + mean
            frame_array: FrameArray = frame_tensor.permute(1, 2, 0).cpu().numpy()
            frame_array = np.clip(frame_array * 255, 0, 255).astype(np.uint8)

            heatmap_resized: np.ndarray = cv2.resize(heatmap, (frame_array.shape[1], frame_array.shape[0]))
            heatmap_colored: np.ndarray = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay: FrameArray = cv2.addWeighted(frame_array, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays

    def save_visualization(self, output_dir: str | Path, num_samples: int = 5) -> list[Path]:
        """Save Grad-CAM grid images for a subset of validation videos."""
        visualization_dir: Path = Path(output_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)

        if self.config.DATASET_NAME == "Mix":
            violence_paths, non_violence_paths = self.config.get_mix_paths()
        else:
            violence_paths = self.config.VIOLENCE_PATH
            non_violence_paths = self.config.NON_VIOLENCE_PATH

        dataset: VideoSequenceDataset = VideoSequenceDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
        )

        sample_count: int = min(num_samples, len(dataset))
        saved_paths: list[Path] = []

        for index in range(sample_count):
            sequence: torch.Tensor
            label: torch.Tensor
            sequence, label = dataset[index]
            heatmap: np.ndarray | None
            predicted_class: int
            probabilities: np.ndarray
            heatmap, predicted_class, probabilities = self.generate_heatmap_for_sequence(sequence)

            if heatmap is None:
                continue

            overlays: list[FrameArray] = self.visualize_heatmap_on_sequence(sequence, heatmap)
            figure, axes = plt.subplots(4, 4, figsize=(16, 16))
            flattened_axes: np.ndarray = axes.flatten()

            for frame_index, overlay in enumerate(overlays):
                if frame_index < len(flattened_axes):
                    flattened_axes[frame_index].imshow(overlay)
                    flattened_axes[frame_index].axis("off")
                    flattened_axes[frame_index].set_title(f"Frame {frame_index + 1}")

            for frame_index in range(len(overlays), len(flattened_axes)):
                flattened_axes[frame_index].axis("off")

            predicted_label: str = "Violence" if predicted_class == 1 else "Non-Violence"
            true_label: str = "Violence" if int(label.item()) == 1 else "Non-Violence"
            confidence: float = float(probabilities[predicted_class] * 100)
            figure.suptitle(f"Prediction: {predicted_label} ({confidence:.1f}%) | True label: {true_label}", fontsize=16)
            figure.tight_layout()

            output_path: Path = visualization_dir / f"sequence_{index}_heatmap.png"
            figure.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(figure)
            saved_paths.append(output_path)

        return saved_paths


def _load_evaluation_model(model_path: str | Path, config: R3DTransferConfig) -> tuple[R3D18Violence, torch.device] | None:
    """Load a trained model checkpoint for evaluation."""
    device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model: R3D18Violence = R3D18Violence(num_classes=2, pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception:
        return None

    model.eval()
    return model, device


def _resolve_dataset_paths(config: R3DTransferConfig) -> tuple[DatasetPath, DatasetPath]:
    """Return the validation source paths configured for the selected dataset."""
    if config.DATASET_NAME == "Mix":
        return config.get_mix_paths()
    return config.VIOLENCE_PATH, config.NON_VIOLENCE_PATH


def _evaluate_dataset(
    model: R3D18Violence,
    device: torch.device,
    dataset: MultiViewVideoDataset,
    config: R3DTransferConfig,
    include_json: bool = False,
) -> tuple[float, list[int], list[int], list[float], list[PredictionJson]]:
    """Run multi-view inference and collect accuracy, labels, probabilities, and optional JSON entries."""
    correct: int = 0
    total: int = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    json_predictions: list[PredictionJson] = []

    with torch.no_grad():
        for index, (clips, label) in enumerate(dataset):
            video_path: Path = dataset.video_paths[index]
            video_name: str = video_path.stem
            clips = clips.to(device)
            label = label.to(device)
            clip_outputs: list[torch.Tensor] = []
            clip_predictions: list[PredictionJson] = []

            for clip_index, clip in enumerate(clips):
                clip_tensor: torch.Tensor = clip.unsqueeze(0)
                output: torch.Tensor = model(clip_tensor)
                clip_outputs.append(output)

                if include_json:
                    clip_probabilities: np.ndarray = torch.softmax(output, dim=1)[0].cpu().numpy()
                    clip_predictions.append(
                        {
                            "algorithmId": "r3d18_violence_detection",
                            "predictions": {
                                "type": "identification",
                                "metadata": {
                                    "video_name": video_name,
                                    "clip_number": clip_index,
                                    "timestamp": clip_index * config.N_FRAMES / 30.0,
                                    "bbox": [0.0, 0.0, 0.0, 0.0],
                                },
                                "class": ["Non-Violent", "Violent"],
                                "score": [float(clip_probabilities[0]), float(clip_probabilities[1])],
                            },
                        }
                    )

            if not clip_outputs:
                continue

            max_output: torch.Tensor = torch.max(torch.stack(clip_outputs), dim=0)[0]
            probabilities: torch.Tensor = torch.softmax(max_output, dim=1)
            predicted: torch.Tensor = torch.max(max_output, 1)[1]
            predicted_value: int = int(predicted.cpu().numpy()[0])
            label_value: int = int(label.cpu().numpy())
            positive_probability: float = float(probabilities[0, 1].cpu().numpy())

            total += 1
            correct += int((predicted == label).sum().item())
            all_predictions.append(predicted_value)
            all_labels.append(label_value)
            all_probabilities.append(positive_probability)

            if include_json:
                json_predictions.extend(clip_predictions)
                max_probabilities: np.ndarray = probabilities[0].cpu().numpy()
                json_predictions.append(
                    {
                        "algorithmId": "r3d18_violence_detection",
                        "predictions": {
                            "type": "identification",
                            "metadata": {
                                "video_name": video_name,
                                "clip_number": "max",
                                "timestamp": 0.0,
                                "bbox": [0.0, 0.0, 0.0, 0.0],
                            },
                            "class": ["Non-Violent", "Violent"],
                            "score": [float(max_probabilities[0]), float(max_probabilities[1])],
                        },
                    }
                )

    accuracy: float = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, all_predictions, all_labels, all_probabilities, json_predictions


def evaluate_model_multiview(
    model_path: str | Path,
    config: R3DTransferConfig,
    num_clips: int = 10,
) -> tuple[float, list[int], list[int], list[float]]:
    """Evaluate a checkpoint with multi-view temporal clips and return raw metrics."""
    loaded_model: tuple[R3D18Violence, torch.device] | None = _load_evaluation_model(model_path, config)
    if loaded_model is None:
        return 0.0, [], [], []

    model: R3D18Violence
    device: torch.device
    model, device = loaded_model
    violence_paths: DatasetPath
    non_violence_paths: DatasetPath
    violence_paths, non_violence_paths = _resolve_dataset_paths(config)

    if violence_paths is None or non_violence_paths is None:
        return 0.0, [], [], []

    val_dataset: MultiViewVideoDataset = MultiViewVideoDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        num_clips=num_clips,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD,
    )

    if len(val_dataset) == 0:
        return 0.0, [], [], []

    accuracy: float
    predictions: list[int]
    labels: list[int]
    probabilities: list[float]
    accuracy, predictions, labels, probabilities, _ = _evaluate_dataset(model, device, val_dataset, config)

    if labels and len(set(labels)) > 1:
        _ = roc_auc_score(labels, probabilities)
        _ = confusion_matrix(labels, predictions)

    return accuracy, predictions, labels, probabilities


def evaluate_model_multiview_with_json(
    model_path: str | Path,
    config: R3DTransferConfig,
    num_clips: int = 10,
) -> tuple[float, list[int], list[int], list[float], list[PredictionJson]]:
    """Evaluate a checkpoint and save per-clip JSON predictions."""
    loaded_model: tuple[R3D18Violence, torch.device] | None = _load_evaluation_model(model_path, config)
    if loaded_model is None:
        return 0.0, [], [], [], []

    model: R3D18Violence
    device: torch.device
    model, device = loaded_model
    violence_paths: DatasetPath
    non_violence_paths: DatasetPath
    violence_paths, non_violence_paths = _resolve_dataset_paths(config)

    if violence_paths is None or non_violence_paths is None:
        return 0.0, [], [], [], []

    val_dataset: MultiViewVideoDataset = MultiViewVideoDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        num_clips=num_clips,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD,
    )

    if len(val_dataset) == 0:
        return 0.0, [], [], [], []

    accuracy: float
    predictions: list[int]
    labels: list[int]
    probabilities: list[float]
    json_predictions: list[PredictionJson]
    accuracy, predictions, labels, probabilities, json_predictions = _evaluate_dataset(
        model,
        device,
        val_dataset,
        config,
        include_json=True,
    )

    results_dir: Path = Path("./results")
    results_dir.mkdir(exist_ok=True, parents=True)
    json_path: Path = results_dir / f"results_{config.DATASET_NAME.lower()}_multiview.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_predictions, file, indent=2)

    return accuracy, predictions, labels, probabilities, json_predictions


def main() -> None:
    """Evaluate the best mixed-dataset checkpoint and save Grad-CAM visualizations."""
    config: R3DTransferConfig = R3DTransferConfig(dataset_name="Mix")
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)
    model_path: Path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        return

    evaluate_model_multiview(model_path, config, num_clips=10)
    generator: HeatmapGenerator3D = HeatmapGenerator3D(model_path, config)
    output_dir: Path = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=5)


if __name__ == "__main__":
    main()