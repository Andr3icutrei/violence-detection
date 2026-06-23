from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import torch

from config import X3DConfig
from evaluate import HeatmapGenerator3DX3D, evaluate_model_multiview
from train import X3DTrainer


logger: logging.Logger = logging.getLogger(__name__)


def count_dataset_videos(config: X3DConfig) -> tuple[int, int, int, int]:
    """Counts violent and non-violent videos and returns train/validation split sizes."""

    if config.VIOLENCE_PATH is None or config.NON_VIOLENCE_PATH is None:
        raise ValueError("Dataset paths must be configured before counting videos.")

    base_path: Path = Path(config.VIOLENCE_PATH["path"])
    violent_videos: list[Path] = []
    non_violent_videos: list[Path] = []

    for dir_name in config.VIOLENCE_PATH["violence_dirs"]:
        dir_path: Path = base_path / dir_name
        if dir_path.exists():
            violent_videos.extend(file_path for file_path in dir_path.rglob("*") if file_path.is_file())

    for dir_name in config.NON_VIOLENCE_PATH["non_violence_dirs"]:
        dir_path = base_path / dir_name
        if dir_path.exists():
            non_violent_videos.extend(file_path for file_path in dir_path.rglob("*") if file_path.is_file())

    violent_train: int = int(len(violent_videos) * config.SPLIT_RATIO)
    violent_val: int = len(violent_videos) - violent_train
    non_violent_train: int = int(len(non_violent_videos) * config.SPLIT_RATIO)
    non_violent_val: int = len(non_violent_videos) - non_violent_train

    return violent_train, violent_val, non_violent_train, non_violent_val


def train_model(config: X3DConfig, verbose: bool = True) -> None:
    """Trains the configured X3D model."""

    trainer: X3DTrainer = X3DTrainer(config, verbose=verbose)
    trainer.train()


def evaluate_trained_model(config: X3DConfig, num_clips: int = 10, save_heatmaps: bool = True) -> None:
    """Evaluates the best saved checkpoint and optionally writes Grad-CAM visualizations."""

    model_path: Path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        logger.error("Model checkpoint not found at %s.", model_path)
        return

    evaluate_model_multiview(model_path, config, num_clips=num_clips)

    if save_heatmaps:
        generator: HeatmapGenerator3DX3D = HeatmapGenerator3DX3D(model_path, config)
        output_dir: Path = Path(f"heatmap_visualizations_x3d_{config.DATASET_NAME.lower()}")
        generator.save_visualization(output_dir, num_samples=10)


def show_dataset_info(config: X3DConfig) -> None:
    """Logs dataset split information."""

    violent_train: int
    violent_val: int
    non_violent_train: int
    non_violent_val: int
    violent_train, violent_val, non_violent_train, non_violent_val = count_dataset_videos(config)

    total_train: int = violent_train + non_violent_train
    total_val: int = violent_val + non_violent_val
    total_videos: int = total_train + total_val

    logger.info(
        "Dataset=%s | total=%d | train=%d | validation=%d | violence_train=%d | violence_validation=%d | non_violence_train=%d | non_violence_validation=%d",
        config.DATASET_NAME,
        total_videos,
        total_train,
        total_val,
        violent_train,
        violent_val,
        non_violent_train,
        non_violent_val,
    )


def build_parser() -> argparse.ArgumentParser:
    """Creates the command-line argument parser."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="X3D violence detection pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "info"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_clips", type=int, default=10)
    parser.add_argument("--no_heatmaps", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def apply_cli_overrides(config: X3DConfig, args: argparse.Namespace) -> None:
    """Applies command-line overrides to the configuration object."""

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        config.EFFECTIVE_BATCH_SIZE = config.BATCH_SIZE * config.ACCUMULATION_STEPS

    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs

    if args.lr is not None:
        config.HEAD_LR = args.lr
        config.BACKBONE_LR = args.lr / 10


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the selected pipeline mode from command-line arguments."""

    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config: X3DConfig = X3DConfig()
    apply_cli_overrides(config, args)

    logger.info(
        "Mode=%s | model=X3D-%s | dataset=%s | device=%s | cuda_available=%s",
        args.mode,
        config.X3D_VERSION.upper(),
        config.DATASET_NAME,
        config.DEVICE,
        torch.cuda.is_available(),
    )

    if args.mode == "train":
        train_model(config, verbose=not args.quiet)
    elif args.mode == "evaluate":
        evaluate_trained_model(config, num_clips=args.num_clips, save_heatmaps=not args.no_heatmaps)
    elif args.mode == "info":
        show_dataset_info(config)


if __name__ == "__main__":
    main()
