from typing import Dict, List, Tuple, Union, Optional
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
import json

def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultiViewSlowFastDataset:
    """Load validation videos as multiple clips for multi-view inference."""

    def __init__(self, violence_path: Union[str, Path, dict], non_violence_path: Union[str, Path, dict], slow_frames: int=8, fast_frames: int=32, temporal_stride: int=2, slowfast_alpha: int=4, slowfast_beta: float=0.125, split_ratio: float=0.8, training: bool=False, num_clips: int=10, mean: List[float]=[0.45, 0.45, 0.45], std: List[float]=[0.225, 0.225, 0.225], crop_size: int=224, seed: int=42, use_crop: bool=False) -> None:
        """Initialize the object and its runtime state."""
        self.slow_frames: int = slow_frames
        self.fast_frames: int = fast_frames
        self.temporal_stride: int = temporal_stride
        self.slowfast_alpha: int = slowfast_alpha
        self.slowfast_beta: float = slowfast_beta
        self.split_ratio: float = split_ratio
        self.training: bool = training
        self.num_clips: int = num_clips
        self.crop_size: int = crop_size
        self.seed: int = seed
        self.use_crop: bool = use_crop
        self._split_rng = random.Random(seed)
        self.mean: torch.Tensor = torch.tensor(mean).view(3, 1, 1, 1)
        self.std: torch.Tensor = torch.tensor(std).view(3, 1, 1, 1)
        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type: str = 'multiclass'
            self.base_path: Path = violence_path['path']
            self.violence_dirs: List[str] = violence_path['violence_dirs']
            self.non_violence_dirs: List[str] = violence_path['non_violence_dirs']
        else:
            raise ValueError('Only AI4RiSK multiclass dataset is supported')
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self) -> Tuple[List[Path], List[int]]:
        """Collect video paths and labels for the selected split."""
        all_videos: List[Path] = []
        all_labels: List[int] = []
        base_path: Path = Path(self.base_path)
        all_dirs: List[str] = self.non_violence_dirs + self.violence_dirs
        for dir_name in all_dirs:
            dir_path: Path = base_path / dir_name
            if dir_path.exists():
                dataset_videos: List[Path] = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                self._split_rng.shuffle(dataset_videos)
                split_idx: int = int(len(dataset_videos) * self.split_ratio)
                if self.training:
                    selected_videos = dataset_videos[:split_idx]
                else:
                    selected_videos = dataset_videos[split_idx:]
                label: int = int(dir_name)
                all_videos.extend(selected_videos)
                all_labels.extend([label] * len(selected_videos))
        return (all_videos, all_labels)

    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Read all RGB frames from a video file."""
        cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _extract_consecutive_clips(self, frames: List[np.ndarray]) -> List[List[int]]:
        """Sample consecutive temporal windows for multi-view evaluation."""
        total_frames: int = len(frames)
        fast_window: int = self.fast_frames * self.temporal_stride
        slow_window: int = self.slow_frames * self.temporal_stride * self.slowfast_alpha
        temporal_window: int = max(fast_window, slow_window)
        if total_frames < temporal_window:
            indices: List[int] = [i % total_frames for i in range(temporal_window)]
            return [indices]
        clips: List[List[int]] = []
        if total_frames < temporal_window * self.num_clips:
            step: int = max(1, (total_frames - temporal_window) // (self.num_clips - 1))
            for i in range(self.num_clips):
                start_idx: int = min(i * step, total_frames - temporal_window)
                clip_indices: List[int] = list(range(start_idx, start_idx + temporal_window))
                clips.append(clip_indices)
        else:
            step = (total_frames - temporal_window) // (self.num_clips - 1)
            for i in range(self.num_clips):
                start_idx = i * step
                clip_indices = list(range(start_idx, start_idx + temporal_window))
                clips.append(clip_indices)
        return clips

    def _preprocess_frame(self, frame: np.ndarray, target_size: int=256) -> np.ndarray:
        """Resize and optionally crop a video frame."""
        frame = frame.astype(np.float32) / 255.0
        if self.use_crop:
            h, w = frame.shape[:2]
            scale: float = target_size / min(h, w)
            new_h, new_w = (int(h * scale), int(w * scale))
            frame = cv2.resize(frame, (new_w, new_h))
            h, w = frame.shape[:2]
            top: int = (h - self.crop_size) // 2
            left: int = (w - self.crop_size) // 2
            frame = frame[top:top + self.crop_size, left:left + self.crop_size]
        else:
            frame = cv2.resize(frame, (self.crop_size, self.crop_size))
        return frame

    def __len__(self) -> int:
        """Return the number of available video samples."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """Return one preprocessed video sample and its label."""
        video_path: Path = self.video_paths[idx]
        label: int = self.labels[idx]
        frames: List[np.ndarray] = self._extract_frames(video_path)
        clip_indices_list: List[List[int]] = self._extract_consecutive_clips(frames)
        processed_clips: List[List[torch.Tensor]] = []
        for clip_indices in clip_indices_list:
            slow_stride: int = self.temporal_stride * self.slowfast_alpha
            slow_frame_indices: List[int] = clip_indices[::slow_stride][:self.slow_frames]
            while len(slow_frame_indices) < self.slow_frames:
                slow_frame_indices.append(slow_frame_indices[-1])
            fast_frame_indices: List[int] = clip_indices[::self.temporal_stride][:self.fast_frames]
            while len(fast_frame_indices) < self.fast_frames:
                fast_frame_indices.append(fast_frame_indices[-1])
            slow_frames: List[np.ndarray] = [frames[i] for i in slow_frame_indices]
            fast_frames: List[np.ndarray] = [frames[i] for i in fast_frame_indices]
            slow_processed: List[np.ndarray] = [self._preprocess_frame(frame) for frame in slow_frames]
            fast_processed: List[np.ndarray] = [self._preprocess_frame(frame) for frame in fast_frames]
            slow_sequence: np.ndarray = np.stack(slow_processed, axis=0)
            fast_sequence: np.ndarray = np.stack(fast_processed, axis=0)
            slow_tensor: torch.Tensor = torch.FloatTensor(slow_sequence).permute(3, 0, 1, 2)
            fast_tensor: torch.Tensor = torch.FloatTensor(fast_sequence).permute(3, 0, 1, 2)
            slow_tensor = (slow_tensor - self.mean) / self.std
            fast_tensor = (fast_tensor - self.mean) / self.std
            processed_clips.append([slow_tensor, fast_tensor])
        if len(processed_clips) == 0:
            slow_zero: torch.Tensor = torch.zeros(3, self.slow_frames, self.crop_size, self.crop_size)
            fast_zero: torch.Tensor = torch.zeros(3, self.fast_frames, self.crop_size, self.crop_size)
            processed_clips = [[slow_zero, fast_zero]]
        label_tensor: torch.Tensor = torch.LongTensor([label])[0]
        return (processed_clips, label_tensor)

class HeatmapGeneratorSlowFast:
    """Generate Grad-CAM heatmaps and overlay visualizations for SlowFast predictions."""

    def __init__(self, model_path: Union[str, Path], config: SlowFastConfig) -> None:
        """Initialize the object and its runtime state."""
        self.config: SlowFastConfig = config
        self.device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model: SlowFastViolence = SlowFastViolence(num_classes=config.NUM_CLASSES, pretrained=False, slowfast_alpha=config.SLOWFAST_ALPHA, slowfast_beta=config.SLOWFAST_BETA).to(self.device)
        checkpoint: dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def generate_heatmap_for_sequence(self, slow_frames: torch.Tensor, fast_frames: torch.Tensor) -> Tuple[Optional[np.ndarray], int, np.ndarray]:
        """Generate a Grad-CAM heatmap for one SlowFast input sequence."""
        slow_frames = slow_frames.unsqueeze(0).to(self.device)
        fast_frames = fast_frames.unsqueeze(0).to(self.device)
        slow_frames.requires_grad = True
        fast_frames.requires_grad = True
        outputs: torch.Tensor = self.model([slow_frames, fast_frames], return_cam=True)
        probs: torch.Tensor = torch.softmax(outputs, dim=1)
        pred_class: int = torch.argmax(probs, dim=1).item()
        target_output: torch.Tensor = outputs[0, pred_class]
        target_output.backward()
        fused_cam_tensor: Optional[torch.Tensor] = self.model.get_fused_spatial_cam(pred_class)
        fused_cam: Optional[np.ndarray] = None
        if fused_cam_tensor is not None:
            fused_cam = fused_cam_tensor[0].cpu().numpy()
        else:
            print('WARNING: Grad-CAM heatmap could not be generated for this sequence.')
        return (fused_cam, pred_class, probs[0].detach().cpu().numpy())

    def visualize_heatmap_on_sequence(self, frames: torch.Tensor, heatmap: np.ndarray, alpha: float=0.5) -> List[np.ndarray]:
        """Overlay a heatmap on each frame of a sequence."""
        overlays: List[np.ndarray] = []
        mean: torch.Tensor = torch.tensor(self.config.KINETICS_MEAN).view(3, 1, 1)
        std: torch.Tensor = torch.tensor(self.config.KINETICS_STD).view(3, 1, 1)
        for i in range(frames.size(1)):
            frame_tensor: torch.Tensor = frames[:, i, :, :]
            frame_tensor = frame_tensor * std + mean
            frame: np.ndarray = frame_tensor.permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            heatmap_resized: np.ndarray = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap_colored: np.ndarray = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay: np.ndarray = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)
        return overlays

    def save_visualization(self, output_dir: Union[str, Path], num_samples: int=5) -> None:
        """Save Grad-CAM visualization GIFs for validation samples."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f'SAVING VISUALIZATIONS TO: {output_dir}')
        print(f"{'=' * 60}")
        violence_path: dict = self.config.VIOLENCE_PATH
        non_violence_path: dict = self.config.NON_VIOLENCE_PATH
        try:
            dataset: SlowFastVideoDataset = SlowFastVideoDataset(violence_path=violence_path, non_violence_path=non_violence_path, slow_frames=self.config.SLOW_FRAMES, fast_frames=self.config.FAST_FRAMES, temporal_stride=self.config.TEMPORAL_STRIDE, slowfast_alpha=self.config.SLOWFAST_ALPHA, slowfast_beta=self.config.SLOWFAST_BETA, split_ratio=self.config.SPLIT_RATIO, training=False, augment=False, mean=self.config.KINETICS_MEAN, std=self.config.KINETICS_STD, crop_size=self.config.CROP_SIZE, use_crop=self.config.USE_CROP)
        except Exception as e:
            print(f'ERROR creating dataset: {e}')
            import traceback
            traceback.print_exc()
            return
        print(f'Total validation videos: {len(dataset)}')
        if len(dataset) == 0:
            print('ERROR: No videos in validation set!')
            return
        num_samples = min(num_samples, len(dataset))
        print(f'Processing {num_samples} samples...')
        count_success: int = 0
        count_fail: int = 0
        for idx in range(num_samples):
            try:
                video_name: str = f'video_{idx}'
                if hasattr(dataset, 'video_paths'):
                    original_path: Path = dataset.video_paths[idx]
                    video_name = Path(original_path).stem
                sequence, label = dataset[idx]
                slow_seq: torch.Tensor = sequence[0]
                fast_seq: torch.Tensor = sequence[1]
                heatmap, pred_class, probs = self.generate_heatmap_for_sequence(slow_seq, fast_seq)
                if heatmap is None:
                    print(f'WARNING: Heatmap is None for sample {idx}')
                    count_fail += 1
                    continue
                overlays: List[np.ndarray] = self.visualize_heatmap_on_sequence(slow_seq, heatmap)
                gif_path: Path = output_dir / f'{video_name}_class{label}_pred{pred_class}.gif'
                imageio.mimsave(str(gif_path), overlays, fps=8, loop=0)
                count_success += 1
            except Exception as e:
                print(f'ERROR processing sample {idx}: {e}')
                import traceback
                traceback.print_exc()
                count_fail += 1
                continue
        print(f"\n{'=' * 60}")
        print(f'VISUALIZATION SUMMARY')
        print(f"{'=' * 60}")
        print(f'Successful: {count_success}/{num_samples}')
        print(f'Failed: {count_fail}/{num_samples}')
        print(f'Output directory: {output_dir}')
        print(f"{'=' * 60}\n")

def evaluate_model_multiview(model_path: Union[str, Path], config: SlowFastConfig, num_clips: int=10) -> Tuple[float, List[int], List[int], List[float]]:
    """Evaluate the model using multiple clips per validation video."""
    print(f"\n{'=' * 60}")
    print(f'EVALUATING MODEL WITH MULTI-VIEW (BINARY)')
    print(f"{'=' * 60}")
    device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model: SlowFastViolence = SlowFastViolence(num_classes=config.NUM_CLASSES, pretrained=False, slowfast_alpha=config.SLOWFAST_ALPHA, slowfast_beta=config.SLOWFAST_BETA).to(device)
    try:
        checkpoint: dict = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
    except Exception as e:
        print(f'ERROR loading model: {e}')
        import traceback
        traceback.print_exc()
        return (0, [], [], [])
    model.eval()
    violence_path: dict = config.VIOLENCE_PATH
    non_violence_path: dict = config.NON_VIOLENCE_PATH
    if violence_path is None or non_violence_path is None:
        return (0, [], [], [])
    try:
        val_dataset: MultiViewSlowFastDataset = MultiViewSlowFastDataset(violence_path=violence_path, non_violence_path=non_violence_path, slow_frames=config.SLOW_FRAMES, fast_frames=config.FAST_FRAMES, temporal_stride=config.TEMPORAL_STRIDE, slowfast_alpha=config.SLOWFAST_ALPHA, slowfast_beta=config.SLOWFAST_BETA, split_ratio=config.SPLIT_RATIO, training=False, num_clips=num_clips, mean=config.KINETICS_MEAN, std=config.KINETICS_STD, crop_size=config.CROP_SIZE, seed=config.SEED, use_crop=config.USE_CROP)
    except Exception as e:
        print(f'ERROR creating dataset: {e}')
        import traceback
        traceback.print_exc()
        return (0, [], [], [])
    print(f'Total validation videos: {len(val_dataset)}')
    if len(val_dataset) == 0:
        return (0, [], [], [])
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_violence_probs: List[float] = []
    with torch.no_grad():
        for video_idx, (clips, label) in enumerate(tqdm(val_dataset, desc='Evaluating')):
            raw_label: int = label.item()
            binary_label: int = 0 if raw_label == 0 else 1
            clip_outputs: List[torch.Tensor] = []
            for clip in clips:
                slow_input: torch.Tensor = clip[0].unsqueeze(0).to(device)
                fast_input: torch.Tensor = clip[1].unsqueeze(0).to(device)
                output: torch.Tensor = model([slow_input, fast_input])
                clip_outputs.append(output)
            if len(clip_outputs) == 0:
                continue
            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)
            probs: torch.Tensor = torch.softmax(max_output, dim=1)
            predicted: int = torch.argmax(max_output, dim=1).item()
            all_preds.append(predicted)
            all_labels.append(binary_label)
            all_violence_probs.append(probs[0, 1].cpu().item())
    if len(all_labels) == 0:
        print('ERROR: No videos processed!')
        return (0, [], [], [])
    accuracy: float = accuracy_score(all_labels, all_preds) * 100
    precision: float = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall: float = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1: float = f1_score(all_labels, all_preds, zero_division=0) * 100
    cm: np.ndarray = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity: float = tn / (tn + fp) * 100 if tn + fp > 0 else 0
    npv: float = tn / (tn + fn) * 100 if tn + fn > 0 else 0
    try:
        sns.set(font_scale=1.5)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Violence", "Violence"], yticklabels=["Non-Violence", "Violence"], annot_kws={'size': 35})
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title('Confusion Matrix', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cm_path: Path = Path(config.SAVE_DIR) / 'confusion_matrix.jpg'
        plt.savefig(cm_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        sns.reset_orig()
        print(f'Confusion matrix saved to: {cm_path}')
    except Exception as e:
        print(f'Confusion matrix save error: {e}')
    try:
        roc_auc: float = roc_auc_score(all_labels, all_violence_probs) * 100
        fpr, tpr, _ = roc_curve(all_labels, all_violence_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc / 100:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        roc_curve_path: Path = Path(config.SAVE_DIR) / 'roc_curve.jpg'
        plt.savefig(roc_curve_path, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'ROC curve saved to: {roc_curve_path}')
    except Exception as e:
        roc_auc = None
        print(f'ROC-AUC error: {e}')
    print(f"\n{'=' * 60}")
    print(f'BINARY EVALUATION RESULTS')
    print(f"{'=' * 60}")
    print(f'Total videos: {len(all_labels)}')
    print(f'Accuracy:     {accuracy:.2f}%')
    print(f'Precision:    {precision:.2f}%')
    print(f'Recall:       {recall:.2f}%')
    print(f'F1-Score:     {f1:.2f}%')
    print(f'Specificity:  {specificity:.2f}%')
    print(f'NPV:          {npv:.2f}%')
    if roc_auc is not None:
        print(f'ROC-AUC:      {roc_auc:.2f}%')
    print(f'\nConfusion Matrix:')
    print(f'                 Predicted')
    print(f'                 Non-V  Violence')
    print(f'Actual Non-V     {cm[0, 0]:5d}  {cm[0, 1]:5d}')
    print(f'       Violence  {cm[1, 0]:5d}  {cm[1, 1]:5d}')
    print(f'\nTP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}')
    print(classification_report(all_labels, all_preds, target_names=["Non-Violence", "Violence"], zero_division=0))
    results: dict = {'total_videos': len(all_labels), 'accuracy': round(accuracy, 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'f1_score': round(f1, 4), 'specificity': round(specificity, 4), 'negative_predictive_value': round(npv, 4), 'roc_auc': round(roc_auc, 4) if roc_auc is not None else None, 'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}, 'num_clips_per_video': num_clips, 'checkpoint_epoch': checkpoint.get('epoch', None)}
    results_path: Path = Path(config.SAVE_DIR) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'\nResults saved to: {results_path}')
    return (accuracy, all_preds, all_labels, all_violence_probs)

def main() -> None:
    """Run standalone evaluation with the configured best checkpoint."""
    config: SlowFastConfig = SlowFastConfig()
    set_seed(config.SEED)
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)
    model_path: Path = config.SAVE_DIR / f'{config.MODEL_NAME}_best.pth'
    if not model_path.exists():
        print(f'ERROR: Model not found at {model_path}')
        print('Train the model first with: python main.py --mode train')
        return
    print('Starting multi-view evaluation.')
    accuracy: float
    all_preds: List[int]
    all_labels: List[int]
    all_probs: List[float]
    accuracy, all_preds, all_labels, all_probs = evaluate_model_multiview(model_path=model_path, config=config, num_clips=4)
    generator: HeatmapGeneratorSlowFast = HeatmapGeneratorSlowFast(model_path, config)
    output_dir: Path = Path(f'heatmap_visualizations_slowfast_{config.DATASET_NAME.lower()}')
    generator.save_visualization(output_dir, num_samples=20)
    print('Evaluation complete.')

if __name__ == '__main__':
    main()