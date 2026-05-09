import torch
import numpy as np
import cv2


def _select_frame_indices(total_frames, config):
    temporal_window = config.NUM_FRAMES * config.TEMPORAL_STRIDE

    if total_frames < temporal_window:
        indices = [i % total_frames for i in range(temporal_window)]
    else:
        start_idx = (total_frames - temporal_window) // 2
        indices = list(range(start_idx, start_idx + temporal_window))

    frame_indices = indices[::config.TEMPORAL_STRIDE][:config.NUM_FRAMES]
    while len(frame_indices) < config.NUM_FRAMES:
        frame_indices.append(frame_indices[-1])
    return frame_indices


def select_r3d_frames_for_overlay(frames, config):
    if not frames:
        return []

    selected_frames = []
    for i in _select_frame_indices(len(frames), config):
        selected_frames.append(cv2.resize(frames[i], (config.CROP_SIZE, config.CROP_SIZE)))
    return selected_frames


def prepare_r3d_tensor(frames, config):
    frame_indices = _select_frame_indices(len(frames), config)

    processed_frames = [
        cv2.resize(frames[i].astype(np.float32) / 255.0, (config.CROP_SIZE, config.CROP_SIZE))
        for i in frame_indices
    ]

    tensor = torch.FloatTensor(np.stack(processed_frames, axis=0)).permute(3, 0, 1, 2)
    mean = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)

    return ((tensor - mean) / std).unsqueeze(0)


class InferencePipelineR3D:
    def __init__(self, model_path, config, model_class):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = model_class(num_classes=config.NUM_CLASSES, pretrained=False).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_and_generate_cam(self, input_tensor):
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        outputs = self.model(input_tensor, return_cam=True)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        self.model.zero_grad()
        outputs[0, pred_class].backward()

        cam_2d = self.model.get_spatial_cam(pred_class)
        heatmap = cam_2d[0].cpu().numpy() if cam_2d is not None else None

        return pred_class, probs[0].detach().cpu().numpy(), heatmap

    def overlay_heatmap_on_frames(self, frames, heatmap, alpha=0.4):
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