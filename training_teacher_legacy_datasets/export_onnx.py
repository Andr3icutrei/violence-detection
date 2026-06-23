from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from config import R3DTransferConfig
from model import R3D18Violence


class R3D18ONNXWrapper(nn.Module):
    """Normalize input tensors before forwarding them through the R3D-18 classifier."""

    def __init__(self, original_model: R3D18Violence) -> None:
        """Register normalization buffers and store the wrapped model."""
        super().__init__()
        self.model: R3D18Violence = original_model
        mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1)
        std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize inputs and return classifier logits."""
        normalized_inputs: torch.Tensor = (inputs - self.input_mean) / self.input_std
        logits: torch.Tensor = self.model(normalized_inputs, return_cam=False)
        return logits


def _load_state_dict(checkpoint_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a model state dictionary from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    opset: int,
    dynamic_axes: bool,
    strict: bool,
    input_size: int,
) -> None:
    """Export a trained R3D-18 checkpoint to a self-contained ONNX file."""
    config: R3DTransferConfig = R3DTransferConfig(dataset_name="Mix")
    model: R3D18Violence = R3D18Violence(
        num_classes=2,
        pretrained=False,
        freeze_layers=None,
        dropout_p=config.DROPOUT_P,
    ).to(device)

    state_dict: dict[str, torch.Tensor] = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    wrapped_model: R3D18ONNXWrapper = R3D18ONNXWrapper(model).to(device)
    wrapped_model.eval()

    dummy_input: torch.Tensor = torch.randn(
        1,
        3,
        config.N_FRAMES,
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
        onnx.save_model(onnx_model, str(output_path))
        external_data_file: Path = output_path.with_name(output_path.name + ".data")
        if external_data_file.exists():
            os.remove(external_data_file)
    except Exception as error:
        raise RuntimeError(f"Could not create a self-contained ONNX model: {error}") from error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ONNX export."""
    config: R3DTransferConfig = R3DTransferConfig(dataset_name="Mix")
    default_checkpoint: Path = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    default_output: Path = config.SAVE_DIR / f"{config.MODEL_NAME}.onnx"
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Export an R3D-18 checkpoint to ONNX format.")
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
    parser.add_argument("--input-size", type=int, default=112, help="Spatial input size used during export.")
    return parser.parse_args()


def main() -> None:
    """Validate paths and export the selected checkpoint to ONNX."""
    args: argparse.Namespace = parse_args()
    checkpoint_path: Path = args.checkpoint.resolve()
    output_path: Path = args.output.resolve()

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
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()