from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from config import X3DConfig
from model import X3DViolence


class X3DONNXWrapper(nn.Module):
    """Normalize raw RGB video tensors before forwarding them through the X3D classifier.

    Expected ONNX input shape: [batch, 3, frames, height, width]
    Expected ONNX input range: [0.0, 1.0], matching the dataset pipeline before normalization.
    """

    def __init__(self, original_model: X3DViolence, mean: list[float], std: list[float]) -> None:
        """Register normalization buffers and store the wrapped model."""
        super().__init__()
        self.model: X3DViolence = original_model

        mean_tensor: torch.Tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1, 1)
        std_tensor: torch.Tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("input_mean", mean_tensor)
        self.register_buffer("input_std", std_tensor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize inputs and return classifier logits."""
        normalized_inputs: torch.Tensor = (inputs - self.input_mean) / self.input_std
        logits: torch.Tensor = self.model(normalized_inputs, return_cam=False)
        return logits


def _strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Strip a common prefix from all state-dict keys when present."""
    if not state_dict:
        return state_dict

    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}

    return state_dict


def _load_state_dict(checkpoint_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a model state dictionary from a checkpoint file."""
    checkpoint: Any = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in: {checkpoint_path}")

    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "model.")
    return state_dict


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    opset: int,
    dynamic_axes: bool,
    strict: bool,
    num_frames: int,
    input_size: int,
) -> None:
    """Export a trained X3D checkpoint to a self-contained ONNX file."""
    config: X3DConfig = X3DConfig()

    model: X3DViolence = X3DViolence(
        num_classes=2,
        pretrained=False,
        dropout_p=config.DROPOUT_P,
        x3d_version=config.X3D_VERSION,
    ).to(device)

    state_dict: dict[str, torch.Tensor] = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    wrapped_model: X3DONNXWrapper = X3DONNXWrapper(
        original_model=model,
        mean=config.KINETICS_MEAN,
        std=config.KINETICS_STD,
    ).to(device)
    wrapped_model.eval()

    dummy_input: torch.Tensor = torch.randn(
        1,
        3,
        num_frames,
        input_size,
        input_size,
        device=device,
    )

    input_names: list[str] = ["input"]
    output_names: list[str] = ["logits"]
    dynamic_axes_map: dict[str, dict[int, str]] | None = None

    if dynamic_axes:
        dynamic_axes_map = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_map,
        dynamo=False,
    )

    output_path = Path(output_path)
    try:
        onnx_model: onnx.ModelProto = onnx.load(str(output_path), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(output_path))

        external_data_file: Path = output_path.with_name(output_path.name + ".data")
        if external_data_file.exists():
            os.remove(external_data_file)
    except Exception as error:
        raise RuntimeError(f"Could not create a valid self-contained ONNX model: {error}") from error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ONNX export."""
    config: X3DConfig = X3DConfig()
    default_checkpoint: Path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    default_output: Path = config.SAVE_DIR / f"{config.MODEL_NAME}.onnx"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Export an X3D checkpoint to ONNX format.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Path to the .pth checkpoint.")
    parser.add_argument("--output", type=Path, default=default_output, help="Path where the .onnx file will be written.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for export.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--dynamic-axes", action="store_true", help="Enable a dynamic batch axis in the exported graph.")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict state dictionary loading.")
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES, help="Temporal input length used during export.")
    parser.add_argument("--input-size", type=int, default=config.INPUT_SIZE, help="Spatial input size used during export.")
    return parser.parse_args()


def main() -> None:
    """Validate paths and export the selected checkpoint to ONNX."""
    args: argparse.Namespace = parse_args()
    checkpoint_path: Path = args.checkpoint.resolve()
    output_path: Path = args.output.resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        device=torch.device(args.device),
        opset=args.opset,
        dynamic_axes=args.dynamic_axes,
        strict=not args.no_strict,
        num_frames=args.num_frames,
        input_size=args.input_size,
    )

    print(f"ONNX model exported to: {output_path}")


if __name__ == "__main__":
    main()
