import torch
import numpy as np
import cv2


def _select_frame_indices(total_frames, config):
    fast_seq_len = config.FAST_FRAMES * config.TEMPORAL_STRIDE
    slow_seq_len = config.SLOW_FRAMES * config.TEMPORAL_STRIDE * config.SLOWFAST_ALPHA
    required_len = max(fast_seq_len, slow_seq_len)

    if total_frames < required_len:
        indices = list(range(total_frames))
        last_idx = total_frames - 1
        while len(indices) < required_len:
            indices.append(last_idx)
    else:
        start_idx = (total_frames - required_len) // 2
        indices = list(range(start_idx, start_idx + required_len))

    slow_stride = config.TEMPORAL_STRIDE * config.SLOWFAST_ALPHA
    slow_indices = indices[::slow_stride][:config.SLOW_FRAMES]
    while len(slow_indices) < config.SLOW_FRAMES:
        slow_indices.append(slow_indices[-1])

    fast_indices = indices[::config.TEMPORAL_STRIDE][:config.FAST_FRAMES]
    while len(fast_indices) < config.FAST_FRAMES:
        fast_indices.append(fast_indices[-1])

    return slow_indices, fast_indices


def select_slow_frames_for_overlay(frames, config):
    if not frames:
        return []
    slow_indices, _ = _select_frame_indices(len(frames), config)
    return [frames[i] for i in slow_indices]


def prepare_slowfast_tensors(frames, config):
    slow_indices, fast_indices = _select_frame_indices(len(frames), config)

    slow_frames = [frames[i] for i in slow_indices]
    fast_frames = [frames[i] for i in fast_indices]

    def process_seq(seq):
        seq_np = np.stack(seq, axis=0).astype(np.float32) / 255.0
        tensor = torch.FloatTensor(seq_np).permute(3, 0, 1, 2)
        mean = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)
        return (tensor - mean) / std

    return process_seq(slow_frames).unsqueeze(0), process_seq(fast_frames).unsqueeze(0)


class InferencePipeline:
    def __init__(self, model_path, config, model_class):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = model_class(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            slowfast_alpha=config.SLOWFAST_ALPHA,
            slowfast_beta=config.SLOWFAST_BETA
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_and_generate_cam(self, slow_tensor, fast_tensor):
        slow_tensor = slow_tensor.to(self.device).requires_grad_(True)
        fast_tensor = fast_tensor.to(self.device).requires_grad_(True)

        outputs = self.model([slow_tensor, fast_tensor], return_cam=True)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        target_output = outputs[0, pred_class]
        target_output.backward()

        fused_cam = self.model.get_fused_spatial_cam(pred_class)
        heatmap = fused_cam[0].cpu().numpy() if fused_cam is not None else None

        return pred_class, probs[0].detach().cpu().numpy(), heatmap

    def overlay_heatmap_on_frames(self, frames, heatmap, alpha=0.5):
        if heatmap is None:
            return []

        if not frames:
            return []

        overlays = []
        heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        hm_min = float(np.min(heatmap))
        hm_max = float(np.max(heatmap))
        if hm_max > hm_min:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        for frame in frames:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays