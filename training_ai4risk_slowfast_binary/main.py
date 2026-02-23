import torch
import random
import numpy as np
from pathlib import Path
import argparse

from config import SlowFastConfig
from train import SlowFastTrainer
from evaluate import evaluate_model_multiview, HeatmapGeneratorSlowFast


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(config):
    print("=" * 60)
    print(f"TRAINING SLOWFAST ON {config.DATASET_NAME.upper()} DATASET")
    print(f"MULTICLASS CLASSIFICATION - {config.NUM_CLASSES} CLASSES")
    print("=" * 60)

    base_path = Path(config.VIOLENCE_PATH['path'])

    class_videos = {}
    for i, dir_name in enumerate(['0', '1', '2', '3', '4']):
        dir_path = base_path / dir_name
        if dir_path.exists():
            videos = list(dir_path.rglob('*'))
            class_videos[i] = videos

    print(f"\nDataset Statistics:")
    total_videos = 0
    for i, class_name in enumerate(config.CLASS_NAMES):
        videos = class_videos.get(i, [])
        train_count = int(len(videos) * config.SPLIT_RATIO)
        val_count = len(videos) - train_count
        print(f"  Class {i} ({class_name}): {len(videos)} total -> {train_count} train, {val_count} val")
        total_videos += len(videos)
    print(f"  Total: {total_videos} videos")
    print()

    trainer = SlowFastTrainer(config)
    trainer.train()

    print("\nTraining completed!")


def evaluate_trained_model(config):
    print("=" * 60)
    print(f"EVALUATING MODEL ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    accuracy, preds, labels, probs = evaluate_model_multiview(model_path, config)

    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)

    generator = HeatmapGeneratorSlowFast(model_path, config)
    output_dir = Path(f"heatmap_visualizations_slowfast_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=10)

    print(f"\nVisualizations saved to {output_dir}")


def show_dataset_info(config):
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"\n{config.DATASET_NAME} Dataset - Multiclass ({config.NUM_CLASSES} classes)")
    print("-" * 60)

    base_path = Path(config.VIOLENCE_PATH['path'])

    violence_dirs = config.VIOLENCE_PATH['violence_dirs']
    non_violence_dirs = config.VIOLENCE_PATH['non_violence_dirs']

    non_violence_videos = []
    for dir_name in non_violence_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            non_violence_videos.extend(list(dir_path.rglob('*')))

    violence_videos = []
    for dir_name in violence_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            violence_videos.extend(list(dir_path.rglob('*')))

    class_videos = {
        0: non_violence_videos,
        1: violence_videos
    }

    total_videos = 0
    total_train = 0
    total_val = 0

    for i, class_name in enumerate(config.CLASS_NAMES):
        videos = class_videos.get(i, [])
        train_count = int(len(videos) * config.SPLIT_RATIO)
        val_count = len(videos) - train_count
        print(f"Class {i} ({class_name}): {len(videos)} total -> {train_count} train, {val_count} val")
        total_videos += len(videos)
        total_train += train_count
        total_val += val_count

    print(f"\nTotal: {total_videos} videos")
    print(f"Training set: {total_train} videos ({config.SPLIT_RATIO:.0%})")
    print(f"Validation set: {total_val} videos ({1 - config.SPLIT_RATIO:.0%})")

    print("\nClass Distribution:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        videos = class_videos.get(i, [])
        percentage = 100 * len(videos) / total_videos if total_videos > 0 else 0
        print(f"  {class_name}: {percentage:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='SlowFast Violence Detection Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'info'],
                        help='Mode: train, evaluate, or info')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for head')

    args = parser.parse_args()

    config = SlowFastConfig()
    set_seed(config.SEED)

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.HEAD_LR = args.lr
        config.BACKBONE_LR = args.lr / 10

    print(f"\n{'=' * 60}")
    print(f"CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model: SlowFast")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Task: Multiclass Classification ({config.NUM_CLASSES} classes)")
    print(f"Classes: {', '.join(config.CLASS_NAMES)}")
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Accumulation steps: {config.ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.EFFECTIVE_BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Slow Frames: {config.SLOW_FRAMES}, Fast Frames: {config.FAST_FRAMES}")
    print(f"Temporal stride: {config.TEMPORAL_STRIDE}")
    print(f"Alpha: {config.SLOWFAST_ALPHA}, Beta: {config.SLOWFAST_BETA}")
    print(f"Backbone LR: {config.BACKBONE_LR}")
    print(f"Head LR: {config.HEAD_LR}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Dropout: {config.DROPOUT_P}")
    print(f"Label Smoothing: {config.LABEL_SMOOTHING}")
    print(f"Gradient Clipping: {config.GRAD_CLIP}")
    if config.USE_FOCAL_LOSS:
        print(f"Loss: Focal Loss (gamma={config.FOCAL_GAMMA})")
    else:
        print(f"Loss: Cross Entropy")
    print(f"Class Weights: {'Enabled' if config.USE_CLASS_WEIGHTS else 'Disabled'}")
    print(f"Balanced Sampling: {'Enabled' if config.USE_BALANCED_SAMPLING else 'Disabled'}")
    print(f"AMP: {'Enabled' if config.USE_AMP else 'Disabled'}")
    if config.USE_SCHEDULER:
        print(f"Scheduler: {config.SCHEDULER_TYPE}")
    print(f"{'=' * 60}\n")

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        evaluate_trained_model(config)
    elif args.mode == 'info':
        show_dataset_info(config)


if __name__ == "__main__":
    main()