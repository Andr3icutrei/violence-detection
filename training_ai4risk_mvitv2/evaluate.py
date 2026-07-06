import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MViTConfig
from dataset import MViTVideoDataset
from model import MViTViolence

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from train import _print_epoch_metrics

def make_soft_mask(patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(patch_size, device=device, dtype=dtype)
    denom = max(patch_size - 1, 1)
    window_1d = 0.5 - 0.5 * torch.cos(2 * np.pi * coords / denom)
    window_2d = torch.outer(window_1d, window_1d)
    window_2d = window_2d / window_2d.max().clamp(min=1e-8)
    return window_2d


def gaussian_blur_video(clip: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    device, dtype = clip.device, clip.dtype
    clip_np = clip.detach().cpu().numpy()
    c, t, h, w = clip_np.shape
    blurred = np.empty_like(clip_np)
    for ti in range(t):
        frame = clip_np[:, ti].transpose(1, 2, 0)  # H, W, C
        frame_blur = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
        if frame_blur.ndim == 2:
            frame_blur = frame_blur[..., None]
        blurred[:, ti] = frame_blur.transpose(2, 0, 1)
    return torch.from_numpy(blurred).to(device=device, dtype=dtype)


def gaussian_blur_map(map2d: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Blur gaussian pe o harta 2D [H, W] (folosit pentru netezirea saliency-ului)."""
    device, dtype = map2d.device, map2d.dtype
    arr = map2d.detach().cpu().numpy().astype(np.float32)
    arr = cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigma)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


class OcclusionSensitivityGenerator:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 mean: List[float], std: List[float]) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(3, 1, 1, 1)
        self.std = torch.tensor(std, device=device).view(3, 1, 1, 1)

    @torch.no_grad()
    def _predict_probs(self, clip_batch: torch.Tensor) -> np.ndarray:
        logits = self.model(clip_batch)
        return F.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def generate(
        self,
        clip: torch.Tensor,
        patch_size: int = 48,
        stride: int = 24,
        t_patch_ratio: float = 0.25,
        t_stride_ratio: float = 0.125,
        batch_size: int = 8,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        clip = clip.to(self.device)
        c, t_len, h, w = clip.shape

        baseline_probs = self._predict_probs(clip.unsqueeze(0))[0]
        pred_class = int(np.argmax(baseline_probs))
        baseline_score = float(baseline_probs[pred_class])

        saliency = torch.zeros((t_len, h, w), device=self.device)
        counts = torch.zeros((t_len, h, w), device=self.device)

        blur_kernel = max(15, (patch_size // 2) | 1)
        blur_sigma = max(3.0, patch_size / 3.0)
        clip_blur = gaussian_blur_video(clip, blur_kernel, blur_sigma)

        soft = make_soft_mask(patch_size, self.device, clip.dtype).view(1, 1, patch_size, patch_size)

        y_positions = list(range(0, max(h - patch_size, 0) + 1, stride))
        if not y_positions or y_positions[-1] != h - patch_size:
            y_positions.append(max(h - patch_size, 0))
        x_positions = list(range(0, max(w - patch_size, 0) + 1, stride))
        if not x_positions or x_positions[-1] != w - patch_size:
            x_positions.append(max(w - patch_size, 0))

        coords = [
            (float(t_frac), y, x)
            for t_frac in np.arange(0.0, 1.0 - t_patch_ratio + 1e-5, t_stride_ratio)
            for y in y_positions
            for x in x_positions
        ]
        t_patch_frames = max(1, int(t_patch_ratio * t_len))

        buf = torch.empty((batch_size, c, t_len, h, w), device=self.device, dtype=clip.dtype)

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            b_size = len(batch_coords)
            patch_ranges = [
                (int(tf * t_len), min(t_len, int(tf * t_len) + t_patch_frames), y, x)
                for tf, y, x in batch_coords
            ]
            b = buf[:b_size]
            b[:] = clip.unsqueeze(0)
            for b_idx, (t_start, t_end, y, x) in enumerate(patch_ranges):
                patch_orig = clip[:, t_start:t_end, y:y + patch_size, x:x + patch_size]
                patch_blur = clip_blur[:, t_start:t_end, y:y + patch_size, x:x + patch_size]
                b[b_idx, :, t_start:t_end, y:y + patch_size, x:x + patch_size] = (
                    patch_orig * (1 - soft) + patch_blur * soft
                )

            probs = self._predict_probs(b)[:, pred_class]
            importances = np.clip(baseline_score - probs, 0.0, None)
            importances_t = torch.from_numpy(importances).to(self.device)

            for b_idx, (t_start, t_end, y, x) in enumerate(patch_ranges):
                saliency[t_start:t_end, y:y + patch_size, x:x + patch_size] += importances_t[b_idx]
                counts[t_start:t_end, y:y + patch_size, x:x + patch_size] += 1.0

        saliency /= torch.clamp(counts, min=1.0)

        for t_idx in range(t_len):
            saliency[t_idx] = gaussian_blur_map(saliency[t_idx], blur_kernel, blur_sigma)

        saliency_np = saliency.cpu().numpy()
        lo, hi = np.percentile(saliency_np, 2), np.percentile(saliency_np, 98)
        if hi - lo > 1e-8:
            saliency_np = np.clip((saliency_np - lo) / (hi - lo), 0, 1)
        else:
            saliency_np = np.zeros_like(saliency_np)

        return pred_class, baseline_probs, saliency_np

    def denormalize(self, clip: torch.Tensor) -> np.ndarray:
        clip = clip.to(self.device) * self.std + self.mean
        clip = clip.clamp(0, 1)
        frames = clip.permute(1, 2, 3, 0).cpu().numpy()
        return (frames * 255).astype(np.uint8)

    @staticmethod
    def make_overlay_frames(frames_uint8: np.ndarray, saliency: np.ndarray, alpha: float = 0.5) -> List[np.ndarray]:
        overlays = []
        for t in range(frames_uint8.shape[0]):
            heat = (saliency[t] * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            frame = frames_uint8[t]
            overlay = cv2.addWeighted(frame, 1 - alpha, heat_color, alpha, 0)
            overlays.append(overlay)
        return overlays

    @staticmethod
    def save_gif(frames: List[np.ndarray], path: Path, fps: int = 8) -> None:
        duration = 1.0 / fps
        imageio.mimsave(str(path), frames, duration=duration, loop=0)

def load_model(config: MViTConfig, device: torch.device) -> torch.nn.Module:
    model = MViTViolence(num_classes=config.NUM_CLASSES, pretrained=False, dropout_p=config.DROPOUT_P)
    ckpt_path = config.SAVE_DIR / f'{config.MODEL_NAME}_best.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint negasit: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    print(f'Model incarcat din {ckpt_path} (epoca {checkpoint.get("epoch", "?")})')
    return model


def build_val_dataset(config: MViTConfig) -> MViTVideoDataset:
    return MViTVideoDataset(
        violence_path=config.VIOLENCE_PATH,
        non_violence_path=config.NON_VIOLENCE_PATH,
        num_frames=config.NUM_FRAMES,
        temporal_stride=config.TEMPORAL_STRIDE,
        split_ratio=config.SPLIT_RATIO,
        training=False,
        augment=False,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD,
        crop_size=config.CROP_SIZE,
        seed=config.SEED,
        use_crop=config.USE_CROP,
    )

def run_evaluation(model: torch.nn.Module, val_dataset: MViTVideoDataset,
                    config: MViTConfig, device: torch.device) -> Tuple[List[int], List[int], List[np.ndarray]]:
    loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluare (sampling centrat)'):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(list(probs))
    _print_epoch_metrics('VAL (centered)', 0.0, all_labels, all_preds, config.CLASS_NAMES)
    return all_labels, all_preds, all_probs

def generate_occlusion_outputs(
    model: torch.nn.Module,
    val_dataset: MViTVideoDataset,
    config: MViTConfig,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 10,
    patch_size: int = 48,
    stride: int = 24,
    t_patch_ratio: float = 0.25,
    t_stride_ratio: float = 0.125,
    occlusion_batch_size: int = 8,
    overlay_alpha: float = 0.5,
    fps: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = OcclusionSensitivityGenerator(model, device, config.KINETICS_MEAN, config.KINETICS_STD)

    num_samples = min(num_samples, len(val_dataset))
    indices = list(range(num_samples))

    for idx in tqdm(indices, desc='Generare GIF-uri occlusion'):
        clip, label = val_dataset[idx]
        video_path = val_dataset.video_paths[idx]

        pred_class, probs, saliency = generator.generate(
            clip,
            patch_size=patch_size,
            stride=stride,
            t_patch_ratio=t_patch_ratio,
            t_stride_ratio=t_stride_ratio,
            batch_size=occlusion_batch_size,
        )

        frames_rgb = generator.denormalize(clip)
        overlay_frames = generator.make_overlay_frames(frames_rgb, saliency, alpha=overlay_alpha)

        label_name = config.CLASS_NAMES[label]
        pred_name = config.CLASS_NAMES[pred_class]
        correct_flag = 'correct' if pred_class == label else 'wrong'
        stem = f'{idx:03d}_{video_path.stem}_gt-{label_name}_pred-{pred_name}_{correct_flag}'

        gif_path = output_dir / f'{stem}.gif'
        generator.save_gif(overlay_frames, gif_path, fps=fps)

        mid_frame = overlay_frames[len(overlay_frames) // 2]
        png_path = output_dir / f'{stem}_mid_heatmap.png'
        cv2.imwrite(str(png_path), cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))

        meta = {
            'video_path': str(video_path),
            'ground_truth': label_name,
            'predicted': pred_name,
            'probs': probs.tolist(),
        }
        with open(output_dir / f'{stem}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

    print(f'\nGIF-uri si heatmap-uri salvate in: {output_dir}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluare MViT (sampling centrat) + occlusion sensitivity GIFs')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'gifs_only'],
                        help="'evaluate' = metrici + gif-uri; 'gifs_only' = sare peste metrici")
    parser.add_argument('--num_gifs', type=int, default=10, help='Numar de clipuri pentru care se genereaza GIF-uri')
    parser.add_argument('--patch_size', type=int, default=48, help='Dimensiunea patch-ului spatial de occlusion')
    parser.add_argument('--stride', type=int, default=24, help='Pasul spatial intre patch-uri')
    parser.add_argument('--t_patch_ratio', type=float, default=0.25, help='Fractia temporala acoperita de un patch')
    parser.add_argument('--t_stride_ratio', type=float, default=0.125, help='Pasul temporal (fractie din clip)')
    parser.add_argument('--occlusion_batch_size', type=int, default=8, help='Batch size pentru forward passes de occlusion')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Transparenta heatmap-ului overlay (0-1)')
    parser.add_argument('--fps', type=int, default=8, help='FPS pentru GIF-urile generate')
    parser.add_argument('--output_dir', type=str, default=None, help='Director de iesire pentru GIF-uri/heatmap-uri')
    args = parser.parse_args()

    config = MViTConfig()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_model(config, device)
    val_dataset = build_val_dataset(config)
    print(f'Clipuri in validation split (sampling centrat): {len(val_dataset)}')

    if args.mode == 'evaluate':
        run_evaluation(model, val_dataset, config, device)

    output_dir = Path(args.output_dir) if args.output_dir else Path(f'occlusion_gifs_{config.MODEL_NAME}')
    generate_occlusion_outputs(
        model, val_dataset, config, device, output_dir,
        num_samples=args.num_gifs,
        patch_size=args.patch_size,
        stride=args.stride,
        t_patch_ratio=args.t_patch_ratio,
        t_stride_ratio=args.t_stride_ratio,
        occlusion_batch_size=args.occlusion_batch_size,
        overlay_alpha=args.overlay_alpha,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()