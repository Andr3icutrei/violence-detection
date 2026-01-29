import torch
from pathlib import Path
import argparse

from config_densenet import DenseNet3DConfig
from train_densenet import DenseNet3DTrainer
from evaluate_densenet import evaluate_model, HeatmapGenerator3DDenseNet


def train_model(config):
    print("=" * 60)
    print(f"TRAINING DENSENET 3D ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()

        for i, (v_path, nv_path) in enumerate(zip(violence_paths, non_violence_paths)):
            dataset_names = ['Crowd', 'Hockey', 'Movies']
            v_videos = list(v_path.rglob('*')) if v_path.exists() else []
            nv_videos = list(nv_path.rglob('*')) if nv_path.exists() else []

            v_train = int(len(v_videos) * config.SPLIT_RATIO)
            v_val = len(v_videos) - v_train
            nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
            nv_val = len(nv_videos) - nv_train

            print(f"{dataset_names[i]}:")
            print(f"  Violence: {len(v_videos)} total -> {v_train} train, {v_val} val")
            print(f"  Non-Violence: {len(nv_videos)} total -> {nv_train} train, {nv_val} val")

        total_violence = sum(len(list(p.rglob('*'))) for p in violence_paths if p.exists())
        total_non_violence = sum(len(list(p.rglob('*'))) for p in non_violence_paths if p.exists())
        total_train = int((total_violence + total_non_violence) * config.SPLIT_RATIO)
        total_val = (total_violence + total_non_violence) - total_train

        print(f"\nTotal across all datasets:")
        print(f"  Train: ~{total_train} videos")
        print(f"  Validation: ~{total_val} videos")

    trainer = DenseNet3DTrainer(config)
    trainer.train()


def evaluate_trained_model(config):
    print("=" * 60)
    print(f"EVALUATING MODEL ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        return

    accuracy, preds, labels, probs = evaluate_model(model_path, config)

    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)

    generator = HeatmapGenerator3DDenseNet(model_path, config)
    output_dir = Path(f"heatmap_visualizations_densenet_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=10)


def show_dataset_info(config):
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()

        total_violence = 0
        total_non_violence = 0
        total_train = 0
        total_val = 0

        for i, (v_path, nv_path) in enumerate(zip(violence_paths, non_violence_paths)):
            dataset_names = ['Crowd', 'Hockey', 'Movies']

            v_videos = list(v_path.rglob('*')) if v_path.exists() else []
            nv_videos = list(nv_path.rglob('*')) if nv_path.exists() else []

            v_train = int(len(v_videos) * config.SPLIT_RATIO)
            v_val = len(v_videos) - v_train
            nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
            nv_val = len(nv_videos) - nv_train

            dataset_train = v_train + nv_train
            dataset_val = v_val + nv_val

            print(f"\n{dataset_names[i]}:")
            print(f"  Violence: {len(v_videos)} total -> {v_train} train, {v_val} val")
            print(f"  Non-Violence: {len(nv_videos)} total -> {nv_train} train, {nv_val} val")
            print(f"  Dataset total: {len(v_videos) + len(nv_videos)} -> {dataset_train} train, {dataset_val} val")

            total_violence += len(v_videos)
            total_non_violence += len(nv_videos)
            total_train += dataset_train
            total_val += dataset_val

        print(f"\n{'=' * 60}")
        print(f"TOTAL COMBINED:")
        print(f"{'=' * 60}")
        print(f"Violence: {total_violence} videos")
        print(f"Non-Violence: {total_non_violence} videos")
        print(f"Total: {total_violence + total_non_violence} videos")
        print(f"Training set: {total_train} videos ({total_train / (total_violence + total_non_violence) * 100:.1f}%)")
        print(f"Validation set: {total_val} videos ({total_val / (total_violence + total_non_violence) * 100:.1f}%)")

    else:
        v_videos = list(config.VIOLENCE_PATH.rglob('*')) if config.VIOLENCE_PATH.exists() else []
        nv_videos = list(config.NON_VIOLENCE_PATH.rglob('*')) if config.NON_VIOLENCE_PATH.exists() else []

        v_train = int(len(v_videos) * config.SPLIT_RATIO)
        v_val = len(v_videos) - v_train
        nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
        nv_val = len(nv_videos) - nv_train

        print(f"\n{config.DATASET_NAME} Dataset")
        print(f"Violence: {len(v_videos)} total -> {v_train} train, {v_val} val")
        print(f"Non-Violence: {len(nv_videos)} total -> {nv_train} train, {nv_val} val")
        print(f"Total: {len(v_videos) + len(nv_videos)} videos")

        total_train = v_train + nv_train
        total_val = v_val + nv_val

        print(f"\nTraining set: {total_train} videos ({config.SPLIT_RATIO:.0%})")
        print(f"Validation set: {total_val} videos ({1 - config.SPLIT_RATIO:.0%})")


def main():
    parser = argparse.ArgumentParser(description='DenseNet 3D Violence Detection Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'info', 'all'],
                        help='Mode: train, evaluate, info, or all')
    parser.add_argument('--dataset', type=str, default='Crowd',
                        choices=['Crowd', 'Hockey', 'Movies', 'Mix'],
                        help='Dataset name: Crowd, Hockey, Movies, or Mix (default: Crowd)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config default)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config default)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--growth_rate', type=int, default=None,
                        help='Growth rate for DenseNet (overrides config default)')

    args = parser.parse_args()

    config = DenseNet3DConfig(dataset_name=args.dataset)

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.growth_rate is not None:
        config.GROWTH_RATE = args.growth_rate

    print(f"\n{'=' * 60}")
    print(f"CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Optimizer: {config.OPTIMIZER.upper()}")
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Growth Rate: {config.GROWTH_RATE}")
    print(f"Block Config: {config.BLOCK_CONFIG}")
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
    elif args.mode == 'all':
        train_model(config)
        evaluate_trained_model(config)


if __name__ == "__main__":
    main()