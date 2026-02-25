import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
import sys
import cv2

sys.path.append('/mnt/project')

from student_model import R3D18Student
from teacher_model import R3D18Violence
from dataset import VideoSequenceDataset
from config import R3DTransferConfig


class StudentEvaluator:
    def __init__(self, student_model_path, teacher_model_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.student_model = R3D18Student(num_classes=2, pretrained=False).to(self.device)
        checkpoint_student = torch.load(student_model_path, map_location=self.device)
        self.student_model.load_state_dict(checkpoint_student['model_state_dict'])
        self.student_model.eval()

        self.teacher_model = R3D18Violence(num_classes=2, pretrained=False).to(self.device)
        checkpoint_teacher = torch.load(teacher_model_path, map_location=self.device)
        self.teacher_model.load_state_dict(checkpoint_teacher['model_state_dict'])
        self.teacher_model.eval()

        self._setup_data()

    def _setup_data(self):
        if self.config.DATASET_NAME == 'Mix':
            violence_paths, non_violence_paths = self.config.get_mix_paths()
        else:
            violence_paths = self.config.VIOLENCE_PATH
            non_violence_paths = self.config.NON_VIOLENCE_PATH

        self.val_dataset = VideoSequenceDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )

    def evaluate(self):
        all_preds_student = []
        all_preds_teacher = []
        all_labels = []
        all_probs_student = []
        all_probs_teacher = []
        all_metadata = []

        criterion = nn.CrossEntropyLoss()
        running_loss_student = 0.0
        running_loss_teacher = 0.0
        correct_student = 0
        correct_teacher = 0
        total = 0

        batch_idx = 0

        for inputs, labels in tqdm(self.val_loader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)

            with torch.no_grad():
                outputs_student = self.student_model(inputs)
                outputs_teacher = self.teacher_model(inputs)

            loss_student = criterion(outputs_student, labels)
            loss_teacher = criterion(outputs_teacher, labels)

            running_loss_student += loss_student.item() * batch_size
            running_loss_teacher += loss_teacher.item() * batch_size

            probs_student = torch.softmax(outputs_student, dim=1)
            probs_teacher = torch.softmax(outputs_teacher, dim=1)

            _, predicted_student = torch.max(outputs_student.data, 1)
            _, predicted_teacher = torch.max(outputs_teacher.data, 1)

            for i in range(batch_size):
                dataset_idx = batch_idx * self.config.BATCH_SIZE + i

                if dataset_idx >= len(self.val_dataset):
                    break

                single_input = inputs[i:i + 1]
                single_input.requires_grad = True

                output_teacher_single = self.teacher_model(single_input, return_cam=True)
                pred_teacher_single = output_teacher_single.argmax(dim=1).item()

                self.teacher_model.zero_grad()
                output_teacher_single[0, pred_teacher_single].backward()

                heatmap = self.teacher_model.get_spatial_cam_plus_plus(pred_teacher_single)
                heatmap_np = heatmap[0].detach().cpu().numpy()

                bbox = self._get_smart_crop_rect(heatmap_np, 112, 112, threshold=self.config.SMART_CROP_THRESHOLD)

                if bbox is None:
                    bbox = [0.0, 0.0, 112.0, 112.0]
                else:
                    bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

                video_info = self.val_dataset.get_video_info(dataset_idx)

                metadata = {
                    'video_name': video_info['video_name'],
                    'sequence_number': video_info['sequence_number'],
                    'timestamp': video_info['timestamp'],
                    'bbox': bbox
                }

                all_metadata.append(metadata)

            total += labels.size(0)
            correct_student += (predicted_student == labels).sum().item()
            correct_teacher += (predicted_teacher == labels).sum().item()

            all_preds_student.extend(predicted_student.cpu().numpy())
            all_preds_teacher.extend(predicted_teacher.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for j in range(batch_size):
                all_probs_student.append(probs_student[j].cpu().numpy())

            all_probs_teacher.extend(probs_teacher[:, 1].cpu().numpy())

            batch_idx += 1

        accuracy_student = 100 * correct_student / total
        accuracy_teacher = 100 * correct_teacher / total
        avg_loss_student = running_loss_student / total
        avg_loss_teacher = running_loss_teacher / total

        try:
            auc_student = roc_auc_score(all_labels, [p[1] for p in all_probs_student])
            auc_teacher = roc_auc_score(all_labels, all_probs_teacher)
        except ValueError:
            auc_student = 0.0
            auc_teacher = 0.0

        cm_student = confusion_matrix(all_labels, all_preds_student)
        cm_teacher = confusion_matrix(all_labels, all_preds_teacher)

        results = {
            'student': {
                'accuracy': accuracy_student,
                'loss': avg_loss_student,
                'auc': auc_student,
                'confusion_matrix': cm_student.tolist()
            },
            'teacher': {
                'accuracy': accuracy_teacher,
                'loss': avg_loss_teacher,
                'auc': auc_teacher,
                'confusion_matrix': cm_teacher.tolist()
            }
        }

        print(f"\nStudent Model (R3D-18 Student):")
        print(f"Accuracy: {accuracy_student:.2f}%")
        print(f"Loss: {avg_loss_student:.4f}")
        print(f"AUC: {auc_student:.4f}")

        print(f"\nTeacher Model (R3D-18):")
        print(f"Accuracy: {accuracy_teacher:.2f}%")
        print(f"Loss: {avg_loss_teacher:.4f}")
        print(f"AUC: {auc_teacher:.4f}")

        return results, all_preds_student, all_preds_teacher, all_labels, all_probs_student, all_metadata

    def save_predictions_json(self, predictions, probabilities, metadata, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_list = []

        for i in range(len(predictions)):
            pred_obj = {
                "algorithmId": "irrelevant",
                "predictions": {
                    "type": "identification",
                    "metadata": {
                        "video_name": metadata[i]['video_name'],
                        "sequence_number": metadata[i]['sequence_number'],
                        "timestamp": metadata[i]['timestamp'],
                        "bbox": metadata[i]['bbox']
                    },
                    "class": ["Non-Violent", "Violent"],
                    "score": [float(probabilities[i][0]), float(probabilities[i][1])]
                }
            }
            predictions_list.append(pred_obj)

        output_path = output_dir / f"results_{self.config.DATASET_NAME.lower()}.json"
        with open(output_path, 'w') as f:
            json.dump(predictions_list, f, indent=2)

        print(f"\nPredictions saved to {output_path}")

    def _get_smart_crop_rect(self, heatmap, original_h, original_w, threshold=0.6):
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
        binary_mask = (heatmap_resized > threshold).astype(np.uint8) * 255

        if binary_mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        new_w = max(w, 30)
        new_h = max(h, 30)

        center_x = x + w // 2
        center_y = y + h // 2

        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)

        if new_x + new_w > original_w: new_x = original_w - new_w
        if new_y + new_h > original_h: new_y = original_h - new_h

        return (max(0, int(new_x)), max(0, int(new_y)), int(new_w), int(new_h))

    def visualize_comparison(self, output_dir, num_samples=10):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_samples = min(num_samples, len(self.val_dataset))
        indices = np.random.choice(len(self.val_dataset), num_samples, replace=False)

        mean = torch.tensor(self.config.KINETICS_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.KINETICS_STD).view(3, 1, 1)

        for idx in tqdm(indices, desc="Generating visualizations"):
            sequence, label = self.val_dataset[idx]

            input_tensor = sequence.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_student = self.student_model(input_tensor)
                probs_student = torch.softmax(output_student, dim=1)[0]
                pred_student = output_student.argmax(dim=1).item()

            input_tensor.requires_grad = True

            output_teacher = self.teacher_model(input_tensor, return_cam=True)
            probs_teacher = torch.softmax(output_teacher, dim=1)[0]
            pred_teacher = output_teacher.argmax(dim=1).item()

            self.teacher_model.zero_grad()
            output_teacher[0, pred_teacher].backward()

            heatmap = self.teacher_model.get_spatial_cam_plus_plus(pred_teacher)
            heatmap_np = heatmap[0].detach().cpu().numpy()

            frames_to_show = [0, 5, 10, 15]
            fig, axes = plt.subplots(2, 4, figsize=(16, 9))

            for i, frame_idx in enumerate(frames_to_show):
                frame_tensor = sequence[:, frame_idx, :, :]
                frame_denorm = frame_tensor * std + mean
                frame_img = frame_denorm.permute(1, 2, 0).cpu().numpy()
                frame_img = np.clip(frame_img, 0, 1)

                h, w = frame_img.shape[:2]

                axes[0, i].imshow(frame_img)
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_title(f"Student Input\nFrame {frame_idx + 1}")
                else:
                    axes[0, i].set_title(f"Frame {frame_idx + 1}")

                heatmap_resized = cv2.resize(heatmap_np, (w, h))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

                overlay = 0.6 * frame_img + 0.4 * heatmap_colored
                overlay = np.clip(overlay, 0, 1)

                bbox = self._get_smart_crop_rect(heatmap_np, h, w, threshold=self.config.SMART_CROP_THRESHOLD)

                if bbox:
                    bx, by, bw, bh = bbox
                    rect = plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='lime', facecolor='none')
                    axes[1, i].add_patch(rect)
                    axes[1, i].text(bx, by - 5, 'Smart Crop', color='lime', fontsize=8, fontweight='bold')

                axes[1, i].imshow(overlay)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_title(f"Teacher GradCAM++ & Crop\nFrame {frame_idx + 1}")

            label_text = "VIOLENCE" if label == 1 else "NON-VIOLENCE"

            student_text = "VIOLENCE" if pred_student == 1 else "NON-VIOLENCE"
            student_conf = probs_student[pred_student].item() * 100

            teacher_text = "VIOLENCE" if pred_teacher == 1 else "NON-VIOLENCE"
            teacher_conf = probs_teacher[pred_teacher].item() * 100

            prob_str_student = f"V: {probs_student[1]:.1%} | NV: {probs_student[0]:.1%}"
            prob_str_teacher = f"V: {probs_teacher[1]:.1%} | NV: {probs_teacher[0]:.1%}"

            fig.suptitle(
                f"True Label: {label_text}\n"
                f"Student: {student_text} ({student_conf:.1f}%) [{prob_str_student}]\n"
                f"Teacher: {teacher_text} ({teacher_conf:.1f}%) [{prob_str_teacher}]",
                fontsize=14, fontweight='bold', y=0.98
            )

            axes[0, 0].text(-20, h // 2, "STUDENT", rotation=90, va='center', fontsize=12, fontweight='bold')
            axes[1, 0].text(-20, h // 2, "TEACHER\n(CAM)", rotation=90, va='center', fontsize=12, fontweight='bold')

            plt.tight_layout()

            output_path = output_dir / f"comparison_{idx}_label_{label.item()}_S{pred_student}_T{pred_teacher}.png"
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()

    def save_results(self, results, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / f"evaluation_results_{self.config.DATASET_NAME.lower()}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_path}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cm_student = np.array(results['student']['confusion_matrix'])
        cm_teacher = np.array(results['teacher']['confusion_matrix'])

        im1 = axes[0].imshow(cm_student, cmap='Blues')
        axes[0].set_title('Student Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Non-Violent', 'Violent'])
        axes[0].set_yticklabels(['Non-Violent', 'Violent'])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm_student[i, j]), ha='center', va='center', color='black')

        im2 = axes[1].imshow(cm_teacher, cmap='Oranges')
        axes[1].set_title('Teacher Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['Non-Violent', 'Violent'])
        axes[1].set_yticklabels(['Non-Violent', 'Violent'])
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, str(cm_teacher[i, j]), ha='center', va='center', color='black')

        plt.tight_layout()
        cm_path = output_dir / "confusion_matrices.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrices saved to {cm_path}")


if __name__ == "__main__":
    config = R3DTransferConfig(dataset_name='Mix', use_smart_crop=True)

    student_model_path = config.get_student_model_path()
    teacher_model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not student_model_path.exists():
        print(f"Student model not found at {student_model_path}")
        sys.exit(1)

    if not teacher_model_path.exists():
        print(f"Teacher model not found at {teacher_model_path}")
        sys.exit(1)

    evaluator = StudentEvaluator(student_model_path, teacher_model_path, config)

    results, preds_student, preds_teacher, labels, probs_student, metadata = evaluator.evaluate()

    output_dir = Path(f"student_evaluation_{config.DATASET_NAME.lower()}")
    evaluator.save_results(results, output_dir)

    evaluator.save_predictions_json(preds_student, probs_student, metadata, output_dir)

    evaluator.visualize_comparison(output_dir, num_samples=10)