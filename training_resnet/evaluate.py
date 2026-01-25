import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from model import R3D18Violence
from dataset import VideoSequenceDataset
from config import R3DTransferConfig


class HeatmapGenerator3D:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = R3D18Violence(num_classes=2, pretrained=False).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_heatmap_for_sequence(self, sequence, target_class=None):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)

        input_tensor = sequence.unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor, return_cam=True)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        cam_2d = self.model.get_spatial_cam(target_class)

        if cam_2d is None:
            return None, target_class, output.softmax(dim=1)[0].cpu().detach().numpy()

        heatmap_2d = cam_2d[0].cpu().numpy()

        return heatmap_2d, target_class, output.softmax(dim=1)[0].cpu().detach().numpy()

    def visualize_heatmap_on_sequence(self, frames, heatmap, alpha=0.4):
        overlays = []

        mean = torch.tensor(self.config.KINETICS_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.KINETICS_STD).view(3, 1, 1)

        for i in range(frames.size(1)):
            frame = frames[:, i, :, :]

            frame = frame * std + mean
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

            heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays

    def save_visualization(self, output_dir, num_samples=5):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # FIX: Gestionare corecta a cailor pentru dataset-ul Mix
        if self.config.DATASET_NAME == 'Mix':
            violence_paths, non_violence_paths = self.config.get_mix_paths()
        else:
            violence_paths = self.config.VIOLENCE_PATH
            non_violence_paths = self.config.NON_VIOLENCE_PATH

        dataset = VideoSequenceDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD
        )

        # Ne asiguram ca nu cerem mai multe sample-uri decat exista in dataset
        num_samples = min(num_samples, len(dataset))

        # Selectam indici aleatori sau primii N
        indices = range(num_samples)

        count = 0
        for idx in indices:
            try:
                sequence, label = dataset[idx]
                heatmap, pred_class, probs = self.generate_heatmap_for_sequence(sequence)

                if heatmap is None:
                    continue

                overlays = self.visualize_heatmap_on_sequence(sequence, heatmap)

                fig, axes = plt.subplots(4, 4, figsize=(16, 16))
                axes = axes.flatten()

                for i, overlay in enumerate(overlays):
                    if i < len(axes):
                        axes[i].imshow(overlay)
                        axes[i].axis('off')
                        axes[i].set_title(f"Frame {i + 1}")

                for i in range(len(overlays), len(axes)):
                    axes[i].axis('off')

                label_text = "Violence" if pred_class == 1 else "Non-Violence"
                true_label = "Violence" if label == 1 else "Non-Violence"
                confidence = probs[pred_class] * 100

                plt.suptitle(f"Pred: {label_text} ({confidence:.1f}%) | True: {true_label}", fontsize=16)
                plt.tight_layout()

                output_path = output_dir / f"sequence_{idx}_heatmap.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                count += 1
                print(f"Saved visualization {count}/{num_samples}")
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue


def evaluate_model(model_path, config):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, [], [], []

    model.eval()

    # FIX: Gestionare corecta a cailor pentru dataset-ul Mix
    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

    print(f"Loading validation dataset for {config.DATASET_NAME}...")

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

    if len(val_dataset) == 0:
        print("Validation dataset is empty! Check paths.")
        return 0, [], [], []

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0

    print("Starting evaluation loop...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    if total == 0:
        print("No samples processed.")
        return 0, [], [], []

    accuracy = 100 * correct / total
    avg_loss = running_loss / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        print("Could not calculate AUC (single class present?)")

    cm = confusion_matrix(all_labels, all_preds)

    # Handle confusing matrix shape safely
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if only one class exists in val set
        tn, fp, fn, tp = 0, 0, 0, 0
        print(f"Confusion Matrix shape unexpected: {cm.shape}")

    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence']))

    return accuracy, all_preds, all_labels, all_probs


def main():
    # Initialize config correctly
    config = R3DTransferConfig(dataset_name="Mix")

    # Ensure save dir exists
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print(f"Current working directory: {Path.cwd()}")
        return

    print("Evaluating R3D-18 Violence Detection Model...")
    accuracy, preds, labels, probs = evaluate_model(model_path, config)

    print("\nGenerating heatmap visualizations...")
    generator = HeatmapGenerator3D(model_path, config)

    # Use a dynamic output folder name based on dataset
    output_dir = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=5)
    print(f"Visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()