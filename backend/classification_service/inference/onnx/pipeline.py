import onnx
import onnxruntime as ort
import torch
import numpy as np
import cv2

from dataclasses import dataclass
from onnx2torch import convert
from typing import Iterable

from inference.onnx.preprocess import OnnxInputSpec


@dataclass(frozen=True)
class OnnxModelMetadata:
    input_specs: list[OnnxInputSpec]
    output_names: list[str]


def _detect_layout(dim_values: list[int | None]) -> str:
    if len(dim_values) != 5:
        return "NCTHW"
    if dim_values[1] == 3:
        return "NCTHW"
    if dim_values[4] == 3:
        return "NTHWC"
    return "NCTHW"


def _extract_input_specs(model: onnx.ModelProto) -> list[OnnxInputSpec]:
    specs: list[OnnxInputSpec] = []
    for input_info in model.graph.input:
        tensor_type = input_info.type.tensor_type
        if not tensor_type or not tensor_type.HasField("shape"):
            continue
        dims = tensor_type.shape.dim
        if len(dims) != 5:
            continue
        dim_values = [d.dim_value if d.dim_value > 0 else None for d in dims]
        layout = _detect_layout(dim_values)
        if layout == "NCTHW":
            num_frames, height, width = dim_values[2], dim_values[3], dim_values[4]
        else:
            num_frames, height, width = dim_values[1], dim_values[2], dim_values[3]
        if not num_frames or not height or not width:
            raise ValueError(
                f"Static input dimensions required for {input_info.name}. Found {dim_values}."
            )
        specs.append(
            OnnxInputSpec(
                name=input_info.name,
                layout=layout,
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
            )
        )
    if not specs:
        raise ValueError("No 5D tensor inputs found in the ONNX model.")
    return specs


def _find_last_conv_output_name(model: onnx.ModelProto) -> str:
    for node in reversed(model.graph.node):
        if node.op_type == "Conv":
            if node.output:
                return node.output[0]
    raise ValueError("No Conv nodes found in ONNX model.")


def _ensure_conv_output(model: onnx.ModelProto, conv_output_name: str) -> onnx.ModelProto:
    if conv_output_name in {out.name for out in model.graph.output}:
        return model
    tensor_info = onnx.helper.make_tensor_value_info(
        conv_output_name,
        onnx.TensorProto.FLOAT,
        None,
    )
    model.graph.output.append(tensor_info)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model


def load_onnx_metadata(model_path: str) -> OnnxModelMetadata:
    import onnx
    model = onnx.load(model_path)
    input_specs = _extract_input_specs(model)
    output_names = [output.name for output in model.graph.output]
    return OnnxModelMetadata(input_specs=input_specs, output_names=output_names)


def gaussian_kernel1d(kernel_size, sigma, device, dtype):
    import torch
    k = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g

def gaussian_blur_video(video, kernel_size=31, sigma=10.0):
    import torch
    import torch.nn.functional as F
    c, t, h, w = video.shape
    kernel_size = max(3, kernel_size | 1)
    g = gaussian_kernel1d(kernel_size, sigma, video.device, video.dtype)
    flat = video.permute(1, 0, 2, 3).reshape(t * c, 1, h, w)
    flat = F.conv2d(flat, g.view(1, 1, 1, -1), padding=(0, kernel_size // 2))
    flat = F.conv2d(flat, g.view(1, 1, -1, 1), padding=(kernel_size // 2, 0))
    return flat.view(t, c, h, w).permute(1, 0, 2, 3)

def gaussian_blur_map(saliency, kernel_size=31, sigma=10.0):
    import torch
    import torch.nn.functional as F
    kernel_size = max(3, kernel_size | 1)
    g = gaussian_kernel1d(kernel_size, sigma, saliency.device, saliency.dtype)
    s = saliency.unsqueeze(0).unsqueeze(0)
    s = F.conv2d(s, g.view(1, 1, 1, -1), padding=(0, kernel_size // 2))
    s = F.conv2d(s, g.view(1, 1, -1, 1), padding=(kernel_size // 2, 0))
    return s[0, 0]

def make_soft_mask(patch_size, device, dtype):
    import torch
    yy, xx = torch.meshgrid(
        torch.arange(patch_size, device=device, dtype=dtype),
        torch.arange(patch_size, device=device, dtype=dtype),
        indexing="ij"
    )
    cy = cx = (patch_size - 1) / 2
    sigma = patch_size / 3.0
    mask = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return mask

class Onnx3dInferencePipeline:
    def __init__(self, model_path: str):
        import onnx
        import onnxruntime as ort
        import torch
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        onnx_model = onnx.load(model_path)
        self.input_specs = _extract_input_specs(onnx_model)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _softmax_numpy(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(
        self,
        inputs: dict[str, np.ndarray],
    ) -> tuple[int, np.ndarray]:
        baseline_inputs = {name: np.expand_dims(val, axis=0) if val.ndim == 4 else val for name, val in inputs.items()}
        
        baseline_outputs = self.session.run(self.output_names, baseline_inputs)[0]
        baseline_probs = self._softmax_numpy(baseline_outputs)
        pred_class = np.argmax(baseline_probs[0])
        return pred_class, baseline_probs[0]

    def predict_and_generate_occlusion(
        self,
        inputs: dict[str, np.ndarray],
        patch_size: int = 48,
        stride: int = 24,
        t_patch_ratio: float = 0.25,
        t_stride_ratio: float = 0.125,
        batch_size: int = 1,
    ) -> tuple[int, np.ndarray, np.ndarray | None]:
        import torch

        baseline_inputs = {name: np.expand_dims(val, axis=0) if val.ndim == 4 else val for name, val in inputs.items()}
        
        baseline_outputs = self.session.run(self.output_names, baseline_inputs)[0]
        baseline_probs = self._softmax_numpy(baseline_outputs)
        pred_class = np.argmax(baseline_probs[0])
        baseline_score = baseline_probs[0, pred_class]

        ref_input_name = self.input_names[0]
        ref_tensor = inputs[ref_input_name]
        
        if ref_tensor.ndim == 5:
            _, c, ref_t_len, h, w = ref_tensor.shape
        else:
            c, ref_t_len, h, w = ref_tensor.shape
            
        saliency = torch.zeros((ref_t_len, h, w), device=self.device)
        counts = torch.zeros((ref_t_len, h, w), device=self.device)

        blur_kernel = max(15, (patch_size // 2) | 1)
        blur_sigma = max(3.0, patch_size / 3.0)
        
        inputs_pt = {}
        inputs_blur = {}
        for name, val in inputs.items():
            if val.ndim == 5:
                val = val[0]
            pt_tensor = torch.from_numpy(val).to(self.device)
            inputs_pt[name] = pt_tensor
            inputs_blur[name] = gaussian_blur_video(pt_tensor, kernel_size=blur_kernel, sigma=blur_sigma)
            
        soft = make_soft_mask(patch_size, self.device, inputs_pt[ref_input_name].dtype).view(1, 1, patch_size, patch_size)

        y_positions = list(range(0, h - patch_size + 1, stride))
        if not y_positions or y_positions[-1] != h - patch_size: y_positions.append(h - patch_size)

        x_positions = list(range(0, w - patch_size + 1, stride))
        if not x_positions or x_positions[-1] != w - patch_size: x_positions.append(w - patch_size)

        coords = []
        for t_frac in np.arange(0.0, 1.0 - t_patch_ratio + 1e-5, t_stride_ratio):
            for y in y_positions:
                for x in x_positions:
                    coords.append((float(t_frac), y, x))

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i: i + batch_size]
            b_size = len(batch_coords)

            batch_inputs_np = {}
            for name, pt_tensor in inputs_pt.items():
                batch_tensor = pt_tensor.unsqueeze(0).repeat(b_size, 1, 1, 1, 1).clone()
                blur_tensor = inputs_blur[name]
                t_len = pt_tensor.shape[1]
                
                for b_idx, (t_frac, y, x) in enumerate(batch_coords):
                    t_start = int(t_frac * t_len)
                    t_end = min(t_len, t_start + max(1, int(t_patch_ratio * t_len)))
                    
                    patch_orig = pt_tensor[:, t_start:t_end, y: y + patch_size, x: x + patch_size]
                    patch_blur = blur_tensor[:, t_start:t_end, y: y + patch_size, x: x + patch_size]
                    
                    # soft is (1, 1, H_p, W_p) but patch is (C, T_p, H_p, W_p)
                    # so numpy/torch broadcasting should work
                    soft_expanded = soft.unsqueeze(1) # (1, 1, 1, H_p, W_p) if soft was (1, 1, H_p, W_p), wait
                    
                    batch_tensor[b_idx, :, t_start:t_end, y: y + patch_size, x: x + patch_size] = \
                        patch_orig * (1 - soft) + patch_blur * soft
                batch_inputs_np[name] = batch_tensor.cpu().numpy()

            outputs = self.session.run(self.output_names, batch_inputs_np)[0]
            scores = self._softmax_numpy(outputs)[:, pred_class]

            importances = baseline_score - scores
            importances = torch.tensor(importances, device=self.device)
            importances = torch.clamp(importances, min=0.0)

            for b_idx, (t_frac, y, x) in enumerate(batch_coords):
                t_start = int(t_frac * ref_t_len)
                t_end = min(ref_t_len, t_start + max(1, int(t_patch_ratio * ref_t_len)))
                
                saliency[t_start:t_end, y: y + patch_size, x: x + patch_size] += importances[b_idx]
                counts[t_start:t_end, y: y + patch_size, x: x + patch_size] += 1.0

        counts = torch.clamp(counts, min=1.0)
        saliency = saliency / counts
        
        # Spatial smooth
        for t_idx in range(ref_t_len):
            saliency[t_idx] = gaussian_blur_map(saliency[t_idx], kernel_size=blur_kernel, sigma=blur_sigma)

        saliency_np = saliency.cpu().numpy()
        lo, hi = np.percentile(saliency_np, 2), np.percentile(saliency_np, 98)
        if hi - lo > 1e-8:
            saliency_np = np.clip((saliency_np - lo) / (hi - lo), 0, 1)
        else:
            saliency_np = np.zeros_like(saliency_np)

        return pred_class, baseline_probs[0], saliency_np

    def overlay_heatmap_on_frames(
        self,
        frames: Iterable[np.ndarray],
        heatmap: np.ndarray | None,
        alpha: float = 0.4,
    ) -> list[np.ndarray]:
        if heatmap is None:
            return []

        frames_list = list(frames)
        num_frames = len(frames_list)
        
        overlays: list[np.ndarray] = []
        heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        is_3d = (heatmap.ndim == 3)
        if not is_3d:
            heatmap = np.expand_dims(heatmap, axis=0).repeat(num_frames, axis=0)
            
        hm_t_len = heatmap.shape[0]

        for i, frame in enumerate(frames_list):
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            t_idx = int((i / num_frames) * hm_t_len)
            t_idx = min(t_idx, hm_t_len - 1)
            hm_slice = heatmap[t_idx]
            
            heatmap_resized = cv2.resize(hm_slice, (frame.shape[1], frame.shape[0]))
            heatmap_colored = cv2.applyColorMap(
                np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays
