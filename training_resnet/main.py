import torch
from pathlib import Path
import argparse

from config import R3DTransferConfig
from train import R3D18Trainer
from evaluate import evaluate_model, evaluate_model_with_json_export, HeatmapGenerator3D
from smart_crop import SmartCropDataset
from torch.utils.data import DataLoader


def train_model(config):
    print("=" * 60)
    print(f"TRAINING R3D-18 ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
        print(f"\nMix dataset includes: Crowd, Hockey, Movies")
        print(
            f"Each dataset contributes {config.SPLIT_RATIO:.0%} to train and {1 - config.SPLIT_RATIO:.0%} to validation")
        print()

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
        print()

    trainer = R3D18Trainer(config)
    trainer.train()

    print("\nTraining completed!")


def evaluate_trained_model(config, max_sequences=3):
    print("=" * 60)
    print(f"EVALUATING MODEL ON {config.DATASET_NAME.upper()} DATASET")
    print("=" * 60)

    if config.use_smart_crop:
        model_path = config.get_smartcrop_model_path(config.DATASET_NAME)
    else:
        model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    accuracy, preds, labels, probs, json_preds = evaluate_model_with_json_export(
        model_path, config, max_sequences=max_sequences
    )

    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)

    generator = HeatmapGenerator3D(model_path, config)
    suffix = "_smartcrop" if config.use_smart_crop else ""
    output_dir = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}{suffix}")
    generator.save_visualization(output_dir, num_samples=10)

    print(f"\nVisualizations saved to {output_dir}")


def test_smart_crop_dataset(config):
    print("=" * 60)
    print("TESTING SMART CROP DATASET")
    print("=" * 60)

    model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train a model first before testing smart crop")
        return

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

    dataset = SmartCropDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        model_path=model_path,
        config=config,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=True,
        use_smart_crop=True,
        smart_crop_prob=0.7
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for i, (sequences, labels) in enumerate(loader):
        print(f"Batch {i + 1}:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Labels: {labels}")

        if i >= 2:
            break

    print("\nSmart crop dataset test completed!")


def train_with_smart_crop(config):
    print("=" * 60)
    print(f"TRAINING WITH SMART CROP ON {config.DATASET_NAME.upper()}")
    print("=" * 60)

    model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not model_path.exists():
        print("No pretrained model found. Train base model first:")
        print(f"  python main.py --mode train --dataset {config.DATASET_NAME}")
        return

    from model import R3D18Violence
    from train import EarlyStopping
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False, dropout_p=config.DROPOUT_P).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

    train_dataset = SmartCropDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        model_path=model_path,
        config=config,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=True,
        use_smart_crop=True,
        smart_crop_prob=0.7
    )

    from dataset import VideoSequenceDataset
    val_dataset = VideoSequenceDataset(
        violence_path=violence_paths,
        non_violence_path=non_violence_paths,
        n_frames=config.N_FRAMES,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        augment=False,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    if config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.BACKBONE_LR,
                                weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.BACKBONE_LR,
                              momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    early_stopping = EarlyStopping(patience=15)
    best_val_loss = float('inf')

    num_epochs = 30

    smartcrop_config = R3DTransferConfig(dataset_name=config.DATASET_NAME, use_smart_crop=True)
    smartcrop_config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training with Smart Crop")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if hasattr(config, 'GRAD_CLIP'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = smartcrop_config.SAVE_DIR / f"{smartcrop_config.MODEL_NAME}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("\nTraining with smart crop completed!")


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

            v_videos = list(v_path.rglob('*')) if v_path.exists() else []
            nv_videos = list(nv_path.rglob('*')) if nv_path.exists() else []

            v_train = int(len(v_videos) * config.SPLIT_RATIO)
            v_val = len(v_videos) - v_train
            nv_train = int(len(nv_videos) * config.SPLIT_RATIO)
            nv_val = len(nv_videos) - nv_train

            dataset_train = v_train + nv_train
            dataset_val = v_val + nv_val

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
        print()
        print(f"Training set: {total_train} videos ({total_train / (total_violence + total_non_violence) * 100:.1f}%)")
        print(f"Validation set: {total_val} videos ({total_val / (total_violence + total_non_violence) * 100:.1f}%)")

    else:
        print(f"\n{config.DATASET_NAME} Dataset")
        print("-" * 60)

        v_videos = list(config.VIOLENCE_PATH.rglob('*')) if config.VIOLENCE_PATH.exists() else []
        nv_videos = list(config.NON_VIOLENCE_PATH.rglob('*')) if config.NON_VIOLENCE_PATH.exists() else []

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
    parser = argparse.ArgumentParser(description='R3D-18 Violence Detection Pipeline with AdamW')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'test_smart_crop', 'train_smart_crop', 'info', 'all'],
                        help='Mode: train, evaluate, test_smart_crop, train_smart_crop, info, or all')
    parser.add_argument('--dataset', type=str, default='Crowd',
                        choices=['Crowd', 'Hockey', 'Movies', 'Mix'],
                        help='Dataset name: Crowd, Hockey, Movies, or Mix (default: Crowd)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config default)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config default)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for head (overrides config default)')
    parser.add_argument('--max_sequences', type=int, default=3,
                        help='Maximum number of sequences to process per video (default: 3)')

    args = parser.parse_args()

    use_smart_crop = args.mode == 'train_smart_crop'
    config = R3DTransferConfig(dataset_name=args.dataset, use_smart_crop=use_smart_crop)

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
    print(f"Max sequences per video: {args.max_sequences}")
    print(f"Model save directory: {config.SAVE_DIR}")
    print(f"{'=' * 60}\n")

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        config_eval = R3DTransferConfig(dataset_name=args.dataset, use_smart_crop=False)
        evaluate_trained_model(config_eval, max_sequences=args.max_sequences)
    elif args.mode == 'test_smart_crop':
        test_smart_crop_dataset(config)
    elif args.mode == 'train_smart_crop':
        train_with_smart_crop(config)
    elif args.mode == 'info':
        show_dataset_info(config)
    elif args.mode == 'all':
        train_model(config)
        evaluate_trained_model(config, max_sequences=args.max_sequences)
        train_with_smart_crop(config)


if __name__ == "__main__":
    main()