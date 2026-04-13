import torch
import random
import cv2
import numpy as np
from pathlib import Path
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

from model import SlowFastViolence
from dataset import SlowFastVideoDataset
from config import SlowFastConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiViewSlowFastDataset:
    def __init__(self, violence_path, non_violence_path,
                 slow_frames=8, fast_frames=32, temporal_stride=2,
                 slowfast_alpha=4, slowfast_beta=0.125,
                 split_ratio=0.8, training=False, num_clips=10,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42, use_crop=False):

        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.temporal_stride = temporal_stride
        self.slowfast_alpha = slowfast_alpha
        self.slowfast_beta = slowfast_beta
        self.split_ratio = split_ratio
        self.training = training
        self.num_clips = num_clips
        self.crop_size = crop_size
        self.seed = seed
        self.use_crop = use_crop
        self._split_rng = random.Random(seed)

        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type = 'multiclass'
            self.base_path = violence_path['path']
            self.violence_dirs = violence_path['violence_dirs']
            self.non_violence_dirs = violence_path['non_violence_dirs']
        else:
            raise ValueError("Only AI4RiSK multiclass dataset is supported")

        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        all_videos = []
        all_labels = []

        base_path = Path(self.base_path)
        all_dirs = self.non_violence_dirs + self.violence_dirs

        for dir_name in all_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                self._split_rng.shuffle(dataset_videos)
                split_idx = int(len(dataset_videos) * self.split_ratio)

                if self.training:
                    selected_videos = dataset_videos[:split_idx]
                else:
                    selected_videos = dataset_videos[split_idx:]

                label = int(dir_name)
                all_videos.extend(selected_videos)
                all_labels.extend([label] * len(selected_videos))

        return all_videos, all_labels

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    def _extract_consecutive_clips(self, frames):
        total_frames = len(frames)
        fast_window = self.fast_frames * self.temporal_stride
        slow_window = self.slow_frames * self.temporal_stride * self.slowfast_alpha
        temporal_window = max(fast_window, slow_window)

        if total_frames < temporal_window:
            indices = [i % total_frames for i in range(temporal_window)]
            return [indices]

        clips = []

        if total_frames < temporal_window * self.num_clips:
            step = max(1, (total_frames - temporal_window) // (self.num_clips - 1))

            for i in range(self.num_clips):
                start_idx = min(i * step, total_frames - temporal_window)
                clip_indices = list(range(start_idx, start_idx + temporal_window))
                clips.append(clip_indices)
        else:
            step = (total_frames - temporal_window) // (self.num_clips - 1)

            for i in range(self.num_clips):
                start_idx = i * step
                clip_indices = list(range(start_idx, start_idx + temporal_window))
                clips.append(clip_indices)

        return clips

    def _preprocess_frame(self, frame, target_size=256):
        frame = frame.astype(np.float32) / 255.0

        if self.use_crop:
            h, w = frame.shape[:2]
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            h, w = frame.shape[:2]
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            frame = frame[top:top + self.crop_size, left:left + self.crop_size]
        else:
            frame = cv2.resize(frame, (self.crop_size, self.crop_size))

        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        clip_indices_list = self._extract_consecutive_clips(frames)

        processed_clips = []

        for clip_indices in clip_indices_list:
            slow_stride = self.temporal_stride * self.slowfast_alpha
            slow_frame_indices = clip_indices[::slow_stride][:self.slow_frames]
            while len(slow_frame_indices) < self.slow_frames:
                slow_frame_indices.append(slow_frame_indices[-1])

            fast_frame_indices = clip_indices[::self.temporal_stride][:self.fast_frames]
            while len(fast_frame_indices) < self.fast_frames:
                fast_frame_indices.append(fast_frame_indices[-1])

            slow_frames = [frames[i] for i in slow_frame_indices]
            fast_frames = [frames[i] for i in fast_frame_indices]

            slow_processed = [self._preprocess_frame(frame) for frame in slow_frames]
            fast_processed = [self._preprocess_frame(frame) for frame in fast_frames]

            slow_sequence = np.stack(slow_processed, axis=0)
            fast_sequence = np.stack(fast_processed, axis=0)

            slow_tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
            fast_tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)

            slow_tensor = (slow_tensor - self.mean) / self.std
            fast_tensor = (fast_tensor - self.mean) / self.std

            processed_clips.append([slow_tensor, fast_tensor])

        if len(processed_clips) == 0:
            slow_zero = torch.zeros(3, self.slow_frames, self.crop_size, self.crop_size)
            fast_zero = torch.zeros(3, self.fast_frames, self.crop_size, self.crop_size)
            processed_clips = [[slow_zero, fast_zero]]

        label = torch.LongTensor([label])[0]

        return processed_clips, label


class HeatmapGeneratorSlowFast:
    def __init__(self, model_path, config):
        print(f"\nInitializing HeatmapGeneratorSlowFast...")
        print(f"  Model path: {model_path}")
        print(f"  Config device: {config.DEVICE}")

        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        print(f"  Creating model...")
        self.model = SlowFastViolence(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            slowfast_alpha=config.SLOWFAST_ALPHA,
            slowfast_beta=config.SLOWFAST_BETA
        ).to(self.device)

        print(f"  Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Model loaded successfully")

        self.model.eval()
        print(f"  Model set to eval mode")

    def generate_heatmap_for_sequence(self, slow_frames, fast_frames):
        slow_frames = slow_frames.unsqueeze(0).to(self.device)
        fast_frames = fast_frames.unsqueeze(0).to(self.device)
        slow_frames.requires_grad = True
        fast_frames.requires_grad = True

        outputs = self.model([slow_frames, fast_frames], return_cam=True)

        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        target_output = outputs[0, pred_class]
        target_output.backward()

        fused_cam = self.model.get_fused_spatial_cam(pred_class)

        if fused_cam is not None:
            fused_cam = fused_cam[0].cpu().numpy()
        else:
            print(f"WARNING: GradCAM returned None")
            print(f"  slow_gradients: {self.model.slow_gradients is not None}")
            print(f"  slow_activations: {self.model.slow_activations is not None}")
            print(f"  fast_gradients: {self.model.fast_gradients is not None}")
            print(f"  fast_activations: {self.model.fast_activations is not None}")

        return fused_cam, pred_class, probs[0].detach().cpu().numpy()

    def visualize_heatmap_on_sequence(self, frames, heatmap, alpha=0.5):
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
        print(f"\n{'=' * 60}")
        print(f"SAVING VISUALIZATIONS TO: {output_dir}")
        print(f"{'=' * 60}")

        violence_path = self.config.VIOLENCE_PATH
        non_violence_path = self.config.NON_VIOLENCE_PATH

        print(f"\nCreating validation dataset...")
        print(f"Violence path type: {type(violence_path)}")
        print(f"Violence path: {violence_path}")

        try:
            dataset = SlowFastVideoDataset(
                violence_path=violence_path,
                non_violence_path=non_violence_path,
                slow_frames=self.config.SLOW_FRAMES,
                fast_frames=self.config.FAST_FRAMES,
                temporal_stride=self.config.TEMPORAL_STRIDE,
                slowfast_alpha=self.config.SLOWFAST_ALPHA,
                slowfast_beta=self.config.SLOWFAST_BETA,
                split_ratio=self.config.SPLIT_RATIO,
                training=False,
                augment=False,
                mean=self.config.KINETICS_MEAN,
                std=self.config.KINETICS_STD,
                crop_size=self.config.CROP_SIZE,
                use_crop=self.config.USE_CROP
            )
        except Exception as e:
            print(f"ERROR creating dataset: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"Dataset created successfully")
        print(f"Total validation videos: {len(dataset)}")

        if len(dataset) == 0:
            print("ERROR: No videos in validation set!")
            return

        num_samples = min(num_samples, len(dataset))
        print(f"Processing {num_samples} samples...")

        count_success = 0
        count_fail = 0

        for idx in range(num_samples):
            print(f"\n--- Processing sample {idx + 1}/{num_samples} ---")
            try:
                video_name = f"video_{idx}"
                if hasattr(dataset, 'video_paths'):
                    original_path = dataset.video_paths[idx]
                    video_name = Path(original_path).stem

                print(f"Processing video: {video_name}")
                print(f"Loading sequence...")
                sequence, label = dataset[idx]
                slow_seq = sequence[0]
                fast_seq = sequence[1]
                print(f"Sequence loaded: slow={slow_seq.shape}, fast={fast_seq.shape}, label={label}")

                print(f"Generating heatmap...")
                heatmap, pred_class, probs = self.generate_heatmap_for_sequence(slow_seq, fast_seq)

                if heatmap is None:
                    print(f"WARNING: Heatmap is None for sample {idx}")
                    count_fail += 1
                    continue

                print(f"Heatmap generated: shape={heatmap.shape}, pred_class={pred_class}")
                print(f"Probabilities: {probs}")

                print(f"Visualizing heatmap on sequence...")
                overlays = self.visualize_heatmap_on_sequence(slow_seq, heatmap)
                print(f"Created {len(overlays)} overlay frames")

                gif_path = output_dir / f"{video_name}_class{label}_pred{pred_class}.gif"
                imageio.mimsave(str(gif_path), overlays, fps=8, loop=0)

                print(f"Saved GIF to: {gif_path}")
                count_success += 1

            except Exception as e:
                print(f"ERROR processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                count_fail += 1
                continue

        print(f"\n{'=' * 60}")
        print(f"VISUALIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Successful: {count_success}/{num_samples}")
        print(f"Failed: {count_fail}/{num_samples}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}\n")


import json

def evaluate_model_multiview(model_path, config, num_clips=10):
    print(f"\n{'=' * 60}")
    print(f"EVALUATING MODEL WITH MULTI-VIEW (BINARY)")
    print(f"{'=' * 60}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SlowFastViolence(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        slowfast_alpha=config.SLOWFAST_ALPHA,
        slowfast_beta=config.SLOWFAST_BETA
    ).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], [], []

    model.eval()

    violence_path = config.VIOLENCE_PATH
    non_violence_path = config.NON_VIOLENCE_PATH

    if violence_path is None or non_violence_path is None:
        return 0, [], [], []

    try:
        val_dataset = MultiViewSlowFastDataset(
            violence_path=violence_path,
            non_violence_path=non_violence_path,
            slow_frames=config.SLOW_FRAMES,
            fast_frames=config.FAST_FRAMES,
            temporal_stride=config.TEMPORAL_STRIDE,
            slowfast_alpha=config.SLOWFAST_ALPHA,
            slowfast_beta=config.SLOWFAST_BETA,
            split_ratio=config.SPLIT_RATIO,
            training=False,
            num_clips=num_clips,
            mean=config.KINETICS_MEAN,
            std=config.KINETICS_STD,
            crop_size=config.CROP_SIZE,
            seed=config.SEED,
            use_crop=config.USE_CROP
        )
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], [], []

    print(f"Total validation videos: {len(val_dataset)}")

    if len(val_dataset) == 0:
        return 0, [], [], []

    all_preds = []
    all_labels = []
    all_violence_probs = []

    with torch.no_grad():
        for video_idx, (clips, label) in enumerate(tqdm(val_dataset, desc="Evaluating")):
            raw_label = label.item()
            binary_label = 0 if raw_label == 0 else 1

            clip_outputs = []
            for clip in clips:
                slow_input = clip[0].unsqueeze(0).to(device)
                fast_input = clip[1].unsqueeze(0).to(device)
                output = model([slow_input, fast_input])
                clip_outputs.append(output)

            if len(clip_outputs) == 0:
                continue

            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)
            probs = torch.softmax(max_output, dim=1)
            predicted = torch.argmax(max_output, dim=1).item()

            all_preds.append(predicted)
            all_labels.append(binary_label)
            all_violence_probs.append(probs[0, 1].cpu().item())

    if len(all_labels) == 0:
        print("ERROR: No videos processed!")
        return 0, [], [], []

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

    try:
        sns.set(font_scale=1.5)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Violence', 'Violence'], 
                    yticklabels=['Non-Violence', 'Violence'],
                    annot_kws={"size": 35})
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title('Confusion Matrix', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cm_path = Path(config.SAVE_DIR) / "confusion_matrix.jpg"
        plt.savefig(cm_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        sns.reset_orig()
        print(f"Confusion matrix saved to: {cm_path}")
    except Exception as e:
        print(f"Confusion matrix save error: {e}")

    try:
        roc_auc = roc_auc_score(all_labels, all_violence_probs) * 100
        
        fpr, tpr, _ = roc_curve(all_labels, all_violence_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc/100:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_curve_path = Path(config.SAVE_DIR) / "roc_curve.jpg"
        plt.savefig(roc_curve_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {roc_curve_path}")
        
    except Exception as e:
        roc_auc = None
        print(f"ROC-AUC error: {e}")

    print(f"\n{'=' * 60}")
    print(f"BINARY EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total videos: {len(all_labels)}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print(f"Precision:    {precision:.2f}%")
    print(f"Recall:       {recall:.2f}%")
    print(f"F1-Score:     {f1:.2f}%")
    print(f"Specificity:  {specificity:.2f}%")
    print(f"NPV:          {npv:.2f}%")
    if roc_auc is not None:
        print(f"ROC-AUC:      {roc_auc:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-V  Violence")
    print(f"Actual Non-V     {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"       Violence  {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    print(f"\nTP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
    print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence'], zero_division=0))

    results = {
        "total_videos": len(all_labels),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "negative_predictive_value": round(npv, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "num_clips_per_video": num_clips,
        "checkpoint_epoch": checkpoint.get('epoch', None)
    }

    results_path = Path(config.SAVE_DIR) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_path}")

    return accuracy, all_preds, all_labels, all_violence_probs


def main():
    print("\n" + "=" * 80)
    print("SLOWFAST VIOLENCE DETECTION - EVALUATION")
    print("=" * 80)

    config = SlowFastConfig()
    set_seed(config.SEED)
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    print(f"\nConfiguration:")
    print(f"  Dataset: {config.DATASET_NAME}")
    print(f"  Num classes: {config.NUM_CLASSES}")
    print(f"  Class names: {config.CLASS_NAMES}")
    print(f"  Device: {config.DEVICE}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Save directory: {config.SAVE_DIR}")
    print(f"  Model name: {config.MODEL_NAME}")

    model_path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    print(f"\nChecking for model at: {model_path}")
    print(f"  Exists: {model_path.exists()}")

    if not model_path.exists():
        print(f"\nERROR: Model not found!")
        print(f"Expected path: {model_path}")
        print(f"Absolute path: {model_path.absolute()}")
        print(f"\nPlease train the model first using:")
        print(f"  python main.py --mode train")
        return

    print(f"  File size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")

    print(f"\n" + "=" * 80)
    print("STEP 1: MULTI-VIEW EVALUATION")
    print("=" * 80)

    accuracy, all_preds, all_labels, all_probs = evaluate_model_multiview(
        model_path=model_path,
        config=config,
        num_clips=4
    )

    generator = HeatmapGeneratorSlowFast(model_path, config)
    output_dir = Path(f"heatmap_visualizations_slowfast_{config.DATASET_NAME.lower()}")
    generator.save_visualization(output_dir, num_samples=20)

    print(f"\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()