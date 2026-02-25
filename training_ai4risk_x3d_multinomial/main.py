import torch
from pathlib import Path
import argparse

from config import X3DConfig
from train import X3DTrainer
from evaluate import evaluate_model_multiview, HeatmapGenerator3DX3D


def train_model(config):
    print("=" * 60)
    print(f"TRAINING X3D-{config.X3D_VERSION.upper()} ON {config.DATASET_NAME.upper()}")
    print("=" * 60)

    base_path = Path(config.DATASET_INFO['path'])
    print(f"\nDataset Statistics:")

    total_videos = 0
    for dir_name in config.DATASET_INFO['dirs']:
        dir_path = base_path / dir_name
        if dir_path.exists():
            videos = list(dir_path.rglob('*'))
            n_train = int(len(videos) * config.SPLIT_RATIO)
            n_val = len(videos) - n_train
            print(f"  Class {dir_name} ({config.CLASS_NAMES[int(dir_name)]}): "
                  f"{len(videos)} total -> {n_train} train, {n_val} val")
            total_videos += len(videos)

    print(f"  Total: {total_videos} videos\n")

    trainer = X3DTrainer(config)
    trainer.train()

    print("\nTraining completed!")


def evaluate_trained_model(config):
    print("=" * 60)
    print(f"EVALUATING MODEL ON {config.DATASET_NAME.upper()}")
    print("=" * 60)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    evaluate_model_multiview(model_path, config)

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
    print(f"\nDataset: {config.DATASET_NAME}")
    print("-" * 60)

    base_path = Path(config.DATASET_INFO['path'])
    total_train, total_val = 0, 0

    for dir_name in config.DATASET_INFO['dirs']:
        dir_path = base_path / dir_name
        if dir_path.exists():
            videos = list(dir_path.rglob('*'))
            n_train = int(len(videos) * config.SPLIT_RATIO)
            n_val = len(videos) - n_train
            total_train += n_train
            total_val += n_val
            print(f"  Class {dir_name} ({config.CLASS_NAMES[int(dir_name)]}): "
                  f"{len(videos)} total -> {n_train} train, {n_val} val")

    print(f"\nTraining set: {total_train} videos ({config.SPLIT_RATIO:.0%})")
    print(f"Validation set: {total_val} videos ({1 - config.SPLIT_RATIO:.0%})")


def main():
    parser = argparse.ArgumentParser(description='X3D Violence Detection Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'info'],
                        help='Mode: train, evaluate, or info')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_clips', type=int, default=None,
                        help='Number of temporal clips for multi-view evaluation')

    args = parser.parse_args()
    config = X3DConfig()

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.HEAD_LR = args.lr
        config.BACKBONE_LR = args.lr / 10
    if args.num_clips is not None:
        config.MULTIVIEW_NUM_CLIPS = args.num_clips

    print(f"\n{'=' * 60}")
    print(f"CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model:              X3D-{config.X3D_VERSION.upper()}")
    print(f"Dataset:            {config.DATASET_NAME}")
    print(f"Classes:            {config.NUM_CLASSES}")
    print(f"Device:             {config.DEVICE}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Batch size:         {config.BATCH_SIZE}")
    print(f"Accumulation steps: {config.ACCUMULATION_STEPS}")
    print(f"Effective batch:    {config.EFFECTIVE_BATCH_SIZE}")
    print(f"Epochs:             {config.NUM_EPOCHS}")
    print(f"Frames:             {config.NUM_FRAMES}, Stride: {config.TEMPORAL_STRIDE}")
    print(f"Backbone LR:        {config.BACKBONE_LR}")
    print(f"Head LR:            {config.HEAD_LR}")
    print(f"Weight Decay:       {config.WEIGHT_DECAY}")
    print(f"Dropout:            {config.DROPOUT_P}")
    print(f"Label Smoothing:    {config.LABEL_SMOOTHING}")
    print(f"Split seed:         {config.SPLIT_SEED}")
    print(f"Best model metric:  {config.BEST_MODEL_METRIC}")
    print(f"Multiview clips:    {config.MULTIVIEW_NUM_CLIPS}")
    if config.USE_SCHEDULER:
        print(f"Scheduler:          {config.SCHEDULER_TYPE}")
    print(f"{'=' * 60}\n")

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        evaluate_trained_model(config)
    elif args.mode == 'info':
        show_dataset_info(config)


if __name__ == "__main__":
    main()