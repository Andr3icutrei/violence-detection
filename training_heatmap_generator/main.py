import torch
from pathlib import Path
import argparse

from config import R3DTransferConfig
from train import R3D18Trainer
from evaluate import evaluate_model, HeatmapGenerator3D
from smart_crop import SmartCropDataset
from torch.utils.data import DataLoader


def train_model(config):
    print("=" * 60)
    print("TRAINING R3D-18 WITH TRANSFER LEARNING")
    print("=" * 60)

    trainer = R3D18Trainer(config)
    trainer.train()

    print("\nTraining completed!")


def evaluate_trained_model(config):
    print("=" * 60)
    print("EVALUATING TRAINED MODEL")
    print("=" * 60)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    accuracy, preds, labels, probs = evaluate_model(model_path, config)

    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)

    generator = HeatmapGenerator3D(model_path, config)
    output_dir = Path("heatmap_visualizations_r3d18")
    generator.save_visualization(output_dir, num_samples=10)

    print(f"\nVisualizations saved to {output_dir}")


def test_smart_crop_dataset(config):
    print("=" * 60)
    print("TESTING SMART CROP DATASET")
    print("=" * 60)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train a model first before testing smart crop")
        return

    dataset = SmartCropDataset(
        violence_path=config.VIOLENCE_PATH,
        non_violence_path=config.NON_VIOLENCE_PATH,
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
    print("TRAINING WITH SMART CROP AUGMENTATION")
    print("=" * 60)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print("No pretrained model found. Train base model first:")
        print("  python main.py --mode train")
        return

    from model import R3D18Violence
    from train import EarlyStopping
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_dataset = SmartCropDataset(
        violence_path=config.VIOLENCE_PATH,
        non_violence_path=config.NON_VIOLENCE_PATH,
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
        violence_path=config.VIOLENCE_PATH,
        non_violence_path=config.NON_VIOLENCE_PATH,
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.BACKBONE_LR,
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    early_stopping = EarlyStopping(patience=15)
    best_val_loss = float('inf')

    num_epochs = 30

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
            save_path = config.SAVE_DIR / f"{config.MODEL_NAME}_smart_crop_best.pth"
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


def main():
    parser = argparse.ArgumentParser(description='R3D-18 Violence Detection Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'test_smart_crop', 'train_smart_crop', 'all'],
                        help='Mode: train, evaluate, test_smart_crop, train_smart_crop, or all')
    parser.add_argument('--dataset', type=str, default='Crowd',
                        help='Dataset name (default: Crowd)')

    args = parser.parse_args()

    config = R3DTransferConfig()
    config.DATASET_NAME = args.dataset
    config.VIOLENCE_PATH = config.DATASET_PATH / config.DATASET_NAME / "Violence"
    config.NON_VIOLENCE_PATH = config.DATASET_PATH / config.DATASET_NAME / "NonViolence"

    print(f"\nDataset: {config.DATASET_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        evaluate_trained_model(config)
    elif args.mode == 'test_smart_crop':
        test_smart_crop_dataset(config)
    elif args.mode == 'train_smart_crop':
        train_with_smart_crop(config)
    elif args.mode == 'all':
        train_model(config)
        evaluate_trained_model(config)
        train_with_smart_crop(config)


if __name__ == "__main__":
    main()