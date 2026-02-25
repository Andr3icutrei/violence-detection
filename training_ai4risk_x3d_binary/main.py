import torch
from pathlib import Path
import argparse

from config import X3DConfig
from train import X3DTrainer
from evaluate import evaluate_model_multiview, HeatmapGenerator3DX3D


def train_model(config):
    print("=" * 60)
    print(f"TRAINING X3D-{config.X3D_VERSION.upper()} ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    base_path = Path(config.VIOLENCE_PATH['path'])

    v_videos = []
    for dir_name in config.VIOLENCE_PATH['violence_dirs']:
        dir_path = base_path / dir_name
        if dir_path.exists():
            v_videos.extend(list(dir_path.rglob('*')))

    nv_videos = []
    for dir_name in config.NON_VIOLENCE_PATH['non_violence_dirs']:
        dir_path = base_path / dir_name
        if dir_path.exists():
            nv_videos.extend(list(dir_path.rglob('*')))

    v_train = int(len(v_videos) * config.SPLIT_RATIO)
    v_val = len(v_videos) - v_train
    nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
    nv_val = len(nv_videos) - nv_train

    print(f"\nDataset Statistics:")
    print(f"  Violence: {len(v_videos)} total -> {v_train} train, {v_val} val")
    print(f"  Non-Violence: {len(nv_videos)} total -> {nv_train} train, {nv_val} val")
    print(f"  Total: {len(v_videos) + len(nv_videos)} videos")
    print()

    trainer = X3DTrainer(config)
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

    generator = HeatmapGenerator3DX3D(model_path, config)
    output_dir = Path(f"heatmap_visualizations_x3d_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=10)

    print(f"\nVisualizations saved to {output_dir}")


def show_dataset_info(config):
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"\n{config.DATASET_NAME} Dataset")
    print("-" * 60)

    base_path = Path(config.VIOLENCE_PATH['path'])

    v_videos = []
    for dir_name in config.VIOLENCE_PATH['violence_dirs']:
        dir_path = base_path / dir_name
        if dir_path.exists():
            v_videos.extend(list(dir_path.rglob('*')))

    nv_videos = []
    for dir_name in config.NON_VIOLENCE_PATH['non_violence_dirs']:
        dir_path = base_path / dir_name
        if dir_path.exists():
            nv_videos.extend(list(dir_path.rglob('*')))

    v_train = int(len(v_videos) * config.SPLIT_RATIO)
    v_val = len(v_videos) - v_train
    nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
    nv_val = len(nv_videos) - nv_train

    print(f"Violence: {len(v_videos)} total -> {v_train} train, {v_val} val")
    print(f"Non-Violence: {len(nv_videos)} total -> {nv_train} train, {nv_val} val")
    print(f"Total: {len(v_videos) + len(nv_videos)} videos")

    total_train = v_train + nv_train
    total_val = v_val + nv_val

    print(f"\nTraining set: {total_train} videos ({config.SPLIT_RATIO:.0%})")
    print(f"Validation set: {total_val} videos ({1 - config.SPLIT_RATIO:.0%})")


def main():
    parser = argparse.ArgumentParser(description='X3D Violence Detection Pipeline')
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

    config = X3DConfig()

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
    print(f"Model: X3D-{config.X3D_VERSION.upper()}")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Accumulation steps: {config.ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.EFFECTIVE_BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Frames: {config.NUM_FRAMES}, Temporal stride: {config.TEMPORAL_STRIDE}")
    print(f"Backbone LR: {config.BACKBONE_LR}")
    print(f"Head LR: {config.HEAD_LR}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Dropout: {config.DROPOUT_P}")
    print(f"Label Smoothing: {config.LABEL_SMOOTHING}")
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