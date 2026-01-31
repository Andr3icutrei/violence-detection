import torch
from pathlib import Path
import argparse

from config import R3DTransferConfig
from train import R3D18Trainer
from evaluate import evaluate_model_multiview_with_json, HeatmapGenerator3D


def train_model(config):
    print("=" * 60)
    print(f"TRAINING R3D-18 ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
        dataset_names = ['Crowd', 'Hockey', 'Movies']
        print(f"\nMix dataset includes: {', '.join(dataset_names)}")
        print(
            f"Each dataset contributes {config.SPLIT_RATIO:.0%} to train and {1 - config.SPLIT_RATIO:.0%} to validation")
        print()

        total_violence = 0
        total_non_violence = 0

        for i, (v_path, nv_path) in enumerate(zip(violence_paths, non_violence_paths)):
            if isinstance(v_path, dict) and v_path.get('type') == 'multiclass':
                base_path = Path(v_path['path'])
                v_videos = []
                for dir_name in v_path['violence_dirs']:
                    dir_path = base_path / dir_name
                    if dir_path.exists():
                        v_videos.extend(list(dir_path.rglob('*')))

                nv_videos = []
                for dir_name in v_path['non_violence_dirs']:
                    dir_path = base_path / dir_name
                    if dir_path.exists():
                        nv_videos.extend(list(dir_path.rglob('*')))
            else:
                v_videos = list(v_path.rglob('*')) if v_path.exists() else []
                nv_videos = list(nv_path.rglob('*')) if nv_path.exists() else []

            v_train = int(len(v_videos) * config.SPLIT_RATIO)
            v_val = len(v_videos) - v_train
            nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
            nv_val = len(nv_videos) - nv_train

            print(f"{dataset_names[i]}:")
            print(f"  Violence: {len(v_videos)} total → {v_train} train, {v_val} val")
            print(f"  Non-Violence: {len(nv_videos)} total → {nv_train} train, {nv_val} val")

            total_violence += len(v_videos)
            total_non_violence += len(nv_videos)

        total_train = int((total_violence + total_non_violence) * config.SPLIT_RATIO)
        total_val = (total_violence + total_non_violence) - total_train

        print(f"\nTotal across all datasets:")
        print(f"  Train: ~{total_train} videos")
        print(f"  Validation: ~{total_val} videos")
        print()

    trainer = R3D18Trainer(config)
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

    accuracy, preds, labels, probs = evaluate_model_multiview_with_json(model_path, config)

    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)

    generator = HeatmapGenerator3D(model_path, config)
    output_dir = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=10)

    print(f"\nVisualizations saved to {output_dir}")


def show_dataset_info(config):
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
        print(f"\nMix Dataset (Crowd + Hockey + Movies)")
        print("=" * 60)
        print(
            f"\nSplit Strategy: Each dataset contributes {config.SPLIT_RATIO:.0%} to train and {1 - config.SPLIT_RATIO:.0%} to validation")
        print("This ensures all datasets are represented in both train and val sets!")
        print("-" * 60)

        total_violence = 0
        total_non_violence = 0
        total_train = 0
        total_val = 0

        for i, (v_path, nv_path) in enumerate(zip(violence_paths, non_violence_paths)):
            dataset_names = ['Crowd', 'Hockey', 'Movies']
            print(f"\n{dataset_names[i]}:")

            if isinstance(v_path, dict) and v_path.get('type') == 'multiclass':
                base_path = Path(v_path['path'])
                v_videos = []
                for dir_name in v_path['violence_dirs']:
                    dir_path = base_path / dir_name
                    if dir_path.exists():
                        v_videos.extend(list(dir_path.rglob('*')))

                nv_videos = []
                for dir_name in v_path['non_violence_dirs']:
                    dir_path = base_path / dir_name
                    if dir_path.exists():
                        nv_videos.extend(list(dir_path.rglob('*')))
            else:
                v_videos = list(v_path.rglob('*')) if v_path.exists() else []
                nv_videos = list(nv_path.rglob('*')) if nv_path.exists() else []

            v_train = int(len(v_videos) * config.SPLIT_RATIO)
            v_val = len(v_videos) - v_train
            nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
            nv_val = len(nv_videos) - nv_train

            dataset_train = v_train + nv_train
            dataset_val = v_val + nv_val

            print(f"  Violence: {len(v_videos)} total → {v_train} train, {v_val} val")
            print(f"  Non-Violence: {len(nv_videos)} total → {nv_train} train, {nv_val} val")
            print(f"  Dataset total: {len(v_videos) + len(nv_videos)} → {dataset_train} train, {dataset_val} val")

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
        print()
        print(f"Training set: {total_train} videos ({total_train / (total_violence + total_non_violence) * 100:.1f}%)")
        print(f"Validation set: {total_val} videos ({total_val / (total_violence + total_non_violence) * 100:.1f}%)")
        print()
        print("✓ All datasets contribute equally to train and val")
        print("✓ No domain shift - model sees all scene types in training")

    else:
        print(f"\n{config.DATASET_NAME} Dataset")
        print("-" * 60)

        if isinstance(config.VIOLENCE_PATH, dict) and config.VIOLENCE_PATH.get('type') == 'multiclass':
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
        else:
            v_videos = list(config.VIOLENCE_PATH.rglob('*')) if config.VIOLENCE_PATH.exists() else []
            nv_videos = list(config.NON_VIOLENCE_PATH.rglob('*')) if config.NON_VIOLENCE_PATH.exists() else []

        v_train = int(len(v_videos) * config.SPLIT_RATIO)
        v_val = len(v_videos) - v_train
        nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
        nv_val = len(nv_videos) - nv_train

        print(f"Violence: {len(v_videos)} total → {v_train} train, {v_val} val")
        print(f"Non-Violence: {len(nv_videos)} total → {nv_train} train, {nv_val} val")
        print(f"Total: {len(v_videos) + len(nv_videos)} videos")

        total_train = v_train + nv_train
        total_val = v_val + nv_val

        print(f"\nTraining set: {total_train} videos ({config.SPLIT_RATIO:.0%})")
        print(f"Validation set: {total_val} videos ({1 - config.SPLIT_RATIO:.0%})")


def main():
    parser = argparse.ArgumentParser(description='R3D-18 Violence Detection Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'info'],
                        help='Mode: train, evaluate, or info')
    parser.add_argument('--dataset', type=str, default='Crowd',
                        choices=['Crowd', 'Hockey', 'Movies', 'RLVS', 'Mix', 'AI4RiSK'],
                        help='Dataset name: Crowd, Hockey, Movies, RLVS, AI4RiSK, or Mix (default: Crowd)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config default)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config default)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for head (overrides config default)')

    args = parser.parse_args()

    config = R3DTransferConfig(dataset_name=args.dataset)

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
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Optimizer: {config.OPTIMIZER.upper()}")
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
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