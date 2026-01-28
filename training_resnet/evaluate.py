import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
import json

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

        num_samples = min(num_samples, len(dataset))

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
            except Exception as e:
                continue


def evaluate_model(model_path, config):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        return 0, [], [], []

    model.eval()

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

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
        return 0, [], [], []

    accuracy = 100 * correct / total
    avg_loss = running_loss / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    return accuracy, all_preds, all_labels, all_probs


def evaluate_model_multiview(model_path, config, num_clips=10):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        return 0, [], [], [], []

    model.eval()

    if config.DATASET_NAME == 'Mix':
        violence_paths, non_violence_paths = config.get_mix_paths()
    else:
        violence_paths = config.VIOLENCE_PATH
        non_violence_paths = config.NON_VIOLENCE_PATH

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
        return 0, [], [], [], []

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    json_predictions = []

    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0

    processed_videos = set()
    mean_tensor = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
    std_tensor = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)

    for idx in tqdm(range(len(val_dataset)), desc="Multi-view Evaluation"):
        video_path = val_dataset.video_paths[idx]
        video_name = video_path.stem
        label = val_dataset.labels[idx]

        if video_name in processed_videos:
            continue

        processed_videos.add(video_name)

        cap = cv2.VideoCapture(str(video_path))
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        cap.release()

        if len(all_frames) < config.N_FRAMES:
            continue

        total_frames = len(all_frames)

        clip_scores = []
        clip_bboxes = []

        effective_clips = min(num_clips, max(1, (total_frames - config.N_FRAMES + 1)))

        if effective_clips == 1:
            start_indices = [0]
        else:
            step = (total_frames - config.N_FRAMES) / (effective_clips - 1)
            start_indices = [int(i * step) for i in range(effective_clips)]

        for clip_idx, start_frame in enumerate(start_indices):
            end_frame = start_frame + config.N_FRAMES

            if end_frame > total_frames:
                break

            sequence_frames = all_frames[start_frame:end_frame]

            processed_frames = []
            for frame in sequence_frames:
                frame_normalized = frame.astype(np.float32) / 255.0
                frame_resized = cv2.resize(frame_normalized, (112, 112))
                processed_frames.append(frame_resized)

            sequence = np.stack(processed_frames, axis=0)
            sequence_tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
            sequence_tensor = (sequence_tensor - mean_tensor) / std_tensor

            input_tensor = sequence_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)

                clip_scores.append(probs[0].cpu().numpy())

            input_tensor.requires_grad = True
            with torch.enable_grad():
                outputs_cam = model(input_tensor, return_cam=True)
                pred_class = outputs_cam.argmax(dim=1).item()

                model.zero_grad()
                outputs_cam[0, pred_class].backward()

                cam_2d = model.get_spatial_cam(pred_class)

                bbox = [0.0, 0.0, 0.0, 0.0]
                if cam_2d is not None:
                    heatmap = cam_2d[0].cpu().numpy()

                    frame_height, frame_width = sequence_frames[0].shape[:2]
                    heatmap_resized = cv2.resize(heatmap, (frame_width, frame_height))

                    threshold = 0.6
                    binary_mask = (heatmap_resized > threshold).astype(np.uint8)

                    if binary_mask.sum() > 0:
                        rows = np.any(binary_mask, axis=1)
                        cols = np.any(binary_mask, axis=0)

                        if rows.any() and cols.any():
                            top = int(np.argmax(rows))
                            bottom = int(len(rows) - np.argmax(rows[::-1]))
                            left = int(np.argmax(cols))
                            right = int(len(cols) - np.argmax(cols[::-1]))

                            bbox = [top, right, bottom, left]

                clip_bboxes.append(bbox)

        clip_scores = np.array(clip_scores)
        avg_probs = np.mean(clip_scores, axis=0)
        max_probs = np.max(clip_scores, axis=0)

        predicted = int(np.argmax(max_probs))

        with torch.no_grad():
            max_probs_tensor = torch.from_numpy(max_probs).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            loss = criterion(max_probs_tensor, label_tensor)
            running_loss += loss.item()

        total += 1
        correct += (predicted == label)

        all_preds.append(predicted)
        all_labels.append(label)
        all_probs.append(float(max_probs[1]))

        for clip_idx in range(len(clip_scores)):
            prediction_entry = {
                "algorithmId": "r3d18_violence_detection",
                "predictions": {
                    "type": "identification",
                    "metadata": {
                        "video_name": video_name,
                        "clip_number": clip_idx,
                        "total_clips": len(clip_scores),
                        "timestamp": float(start_indices[clip_idx] / 30.0),
                        "bbox": clip_bboxes[clip_idx]
                    },
                    "class": ["Non-Violent", "Violent"],
                    "score": [float(clip_scores[clip_idx][0]), float(clip_scores[clip_idx][1])]
                }
            }
            json_predictions.append(prediction_entry)

        aggregated_entry = {
            "algorithmId": "r3d18_violence_detection",
            "predictions": {
                "type": "identification",
                "metadata": {
                    "video_name": video_name,
                    "aggregation": "max",
                    "num_clips": len(clip_scores),
                    "bbox": [0.0, 0.0, 0.0, 0.0]
                },
                "class": ["Non-Violent", "Violent"],
                "score": [float(max_probs[0]), float(max_probs[1])]
            }
        }
        json_predictions.append(aggregated_entry)

    if total == 0:
        return 0, [], [], [], []

    accuracy = 100 * correct / total
    avg_loss = running_loss / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence']))

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True, parents=True)

    json_path = results_dir / f"results_{config.DATASET_NAME.lower()}.json"

    with open(json_path, 'w') as f:
        json.dump(json_predictions, f, indent=2)

    return accuracy, all_preds, all_labels, all_probs, json_predictions


def main():
    config = R3DTransferConfig(dataset_name="Mix")

    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"

    if not model_path.exists():
        return

    accuracy_mv, preds_mv, labels_mv, probs_mv, json_preds = evaluate_model_multiview(
        model_path, config, num_clips=10
    )

    generator = HeatmapGenerator3D(model_path, config)

    output_dir = Path(f"heatmap_visualizations_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=5)


if __name__ == "__main__":
    main()