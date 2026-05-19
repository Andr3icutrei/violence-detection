import onnx
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
        dim_values: list[int | None] = [d.dim_value if d.dim_value > 0 else None for d in dims]
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
    model = onnx.load(model_path)
    input_specs = _extract_input_specs(model)
    output_names = [output.name for output in model.graph.output]
    return OnnxModelMetadata(input_specs=input_specs, output_names=output_names)


class Onnx3dInferencePipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        onnx_model = onnx.load(model_path)
        self.input_specs = _extract_input_specs(onnx_model)
        conv_output_name = _find_last_conv_output_name(onnx_model)
        onnx_model = _ensure_conv_output(onnx_model, conv_output_name)
        self.output_names = [output.name for output in onnx_model.graph.output]
        self.input_names = [input_info.name for input_info in onnx_model.graph.input]

        self.model = convert(onnx_model).to(self.device)
        self.model.eval()

    def _run_forward(self, inputs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        ordered_inputs = [
            torch.from_numpy(inputs[name]).to(self.device)
            for name in self.input_names
            if name in inputs
        ]
        if not ordered_inputs:
            raise ValueError("No inputs were prepared for ONNX inference.")

        outputs = self.model(*ordered_inputs)
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            conv_features = outputs[-1]
        else:
            raise ValueError("Expected ONNX model to return logits and Conv features outputs.")

        if not isinstance(conv_features, torch.Tensor):
            raise ValueError("Conv features output is missing or invalid.")

        conv_features.retain_grad()
        return logits, conv_features

    def predict_and_generate_cam(
        self,
        inputs: dict[str, np.ndarray],
    ) -> tuple[int, np.ndarray, np.ndarray | None]:
        logits, conv_features = self._run_forward(inputs)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        self.model.zero_grad()
        logits[0, pred_class].backward()

        grads = conv_features.grad
        if grads is None:
            return pred_class, probs[0].detach().cpu().numpy(), None

        if conv_features.ndim != 5:
            return pred_class, probs[0].detach().cpu().numpy(), None

        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * conv_features).sum(dim=1)
        cam = torch.relu(cam)
        cam_2d = cam.mean(dim=1)

        cam_2d = cam_2d.detach().cpu().numpy()[0]
        return pred_class, probs[0].detach().cpu().numpy(), cam_2d

    def overlay_heatmap_on_frames(
        self,
        frames: Iterable[np.ndarray],
        heatmap: np.ndarray | None,
        alpha: float = 0.4,
    ) -> list[np.ndarray]:
        if heatmap is None:
            return []

        overlays: list[np.ndarray] = []
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
            heatmap_colored = cv2.applyColorMap(
                np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
            overlays.append(overlay)

        return overlays

