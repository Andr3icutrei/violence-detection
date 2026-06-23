from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import R3DTransferConfig, DatasetInfo
from evaluate import HeatmapGenerator3D, evaluate_model_multiview_with_json, PredictionJson
from train import R3D18Trainer


def _count_videos(path: Path) -> int:
    """Count files recursively under a dataset folder."""
    if not path.exists():
        return 0
    video_count: int = sum(1 for file_path in path.rglob("*") if file_path.is_file())
    return video_count


def _count_multiclass_videos(dataset_info: DatasetInfo, class_dirs_key: str) -> int:
    """Count files in multiclass dataset directories that map to one binary class."""
    base_path: Path = Path(dataset_info["path"])
    total_count: int = 0

    for directory_name in dataset_info[class_dirs_key]:
        directory_path: Path = base_path / directory_name
        total_count += _count_videos(directory_path)

    return total_count


def train_model(config: R3DTransferConfig) -> dict[str, list[float] | list[list[float]]]:
    """Train the configured model and return its metric history."""
    trainer: R3D18Trainer = R3D18Trainer(config)
    history: dict[str, list[float] | list[list[float]]] = trainer.train()
    return history


def evaluate_trained_model(config: R3DTransferConfig) -> dict[str, str | bool | float | int | list[str]]:
    """Evaluate the best checkpoint and save Grad-CAM visualizations."""
    model_path: Path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        return {"model_path": str(model_path), "exists": False}

    accuracy: float
    predictions: list[int]
    labels: list[int]
    probabilities: list[float]
    json_predictions: list[PredictionJson]
    accuracy, predictions, labels, probabilities, json_predictions = evaluate_model_multiview_with_json(model_path, config)

    generator: HeatmapGenerator3D = HeatmapGenerator3D(model_path, config)
    output_dir: Path = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}")
    visualization_paths: list[Path] = generator.save_visualization(output_dir, num_samples=10)

    return {
        "model_path": str(model_path),
        "exists": True,
        "accuracy": accuracy,
        "num_predictions": len(predictions),
        "num_labels": len(labels),
        "num_probabilities": len(probabilities),
        "num_json_predictions": len(json_predictions),
        "visualizations": [str(path) for path in visualization_paths],
    }


def collect_dataset_info(config: R3DTransferConfig) -> dict[str, str | float | list[dict[str, str | int]] | dict[str, int]]:
    """Collect dataset sizes and split counts without starting training or evaluation."""
    dataset_summaries: list[dict[str, str | int]] = []
    total_violence: int = 0
    total_non_violence: int = 0
    total_train: int = 0
    total_validation: int = 0

    if config.DATASET_NAME == "Mix":
        violence_paths, non_violence_paths = config.get_mix_paths()
        dataset_names: list[str] = ["Crowd", "Hockey", "Movies"]
    else:
        violence_paths = [config.VIOLENCE_PATH]
        non_violence_paths = [config.NON_VIOLENCE_PATH]
        dataset_names = [config.DATASET_NAME]

    for dataset_index, (violence_path, non_violence_path) in enumerate(zip(violence_paths, non_violence_paths)):
        dataset_name: str = dataset_names[dataset_index]

        if isinstance(violence_path, dict) and violence_path.get("type") == "multiclass":
            violence_count: int = _count_multiclass_videos(violence_path, "violence_dirs")
            non_violence_count: int = _count_multiclass_videos(violence_path, "non_violence_dirs")
        else:
            violence_count = _count_videos(Path(violence_path))
            non_violence_count = _count_videos(Path(non_violence_path))

        violence_train: int = int(violence_count * config.SPLIT_RATIO)
        violence_validation: int = violence_count - violence_train
        non_violence_train: int = int(non_violence_count * config.SPLIT_RATIO)
        non_violence_validation: int = non_violence_count - non_violence_train
        dataset_train: int = violence_train + non_violence_train
        dataset_validation: int = violence_validation + non_violence_validation

        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "violence_total": violence_count,
                "violence_train": violence_train,
                "violence_validation": violence_validation,
                "non_violence_total": non_violence_count,
                "non_violence_train": non_violence_train,
                "non_violence_validation": non_violence_validation,
                "dataset_total": violence_count + non_violence_count,
                "dataset_train": dataset_train,
                "dataset_validation": dataset_validation,
            }
        )

        total_violence += violence_count
        total_non_violence += non_violence_count
        total_train += dataset_train
        total_validation += dataset_validation

    return {
        "dataset_name": config.DATASET_NAME,
        "split_ratio": config.SPLIT_RATIO,
        "datasets": dataset_summaries,
        "totals": {
            "violence": total_violence,
            "non_violence": total_non_violence,
            "videos": total_violence + total_non_violence,
            "train": total_train,
            "validation": total_validation,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the training and evaluation pipeline."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="R3D-18 violence detection pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "info"], help="Pipeline mode.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Crowd",
        choices=["Crowd", "Hockey", "Movies", "Mix", "AI4RiSK"],
        help="Dataset name.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override.")
    parser.add_argument("--lr", type=float, default=None, help="Head learning-rate override.")
    return parser.parse_args()


def main() -> None:
    """Run the selected pipeline mode."""
    args: argparse.Namespace = parse_args()
    config: R3DTransferConfig = R3DTransferConfig(dataset_name=args.dataset)

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.HEAD_LR = args.lr
        config.BACKBONE_LR = args.lr / 10

    if args.mode == "train":
        train_model(config)
    elif args.mode == "evaluate":
        evaluate_trained_model(config)
    elif args.mode == "info":
        dataset_info = collect_dataset_info(config)
        print(json.dumps(dataset_info, indent=2))


if __name__ == "__main__":
    main()