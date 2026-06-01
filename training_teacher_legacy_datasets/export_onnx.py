import argparse
import os  # <-- Adăugat
from pathlib import Path

import onnx  # <-- Adăugat (asigură-te că ai instalat pachetul cu: pip install onnx)
import torch
import torch.nn as nn

from config import R3DTransferConfig
from model import R3D18Violence


class R3D18ONNXWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        # ImageNet mean/std for 0-1 inputs, broadcastable to N,C,T,H,W
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        return self.model(x, return_cam=False)


def _load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def export_onnx(checkpoint_path, output_path, device, opset, dynamic_axes, strict, input_size):
    config = R3DTransferConfig(dataset_name="Mix")
    model = R3D18Violence(
        num_classes=2,
        pretrained=False,
        freeze_layers=None,
        dropout_p=config.DROPOUT_P,
    ).to(device)

    state_dict = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    wrapped_model = R3D18ONNXWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()  # <-- FIX IMPORTANT: Trecem wrapper-ul în modul de inferență

    dummy_input = torch.randn(
        1,
        3,
        config.N_FRAMES,
        input_size,
        input_size,
        device=device,
    )

    input_names = ["input"]
    output_names = ["logits"]

    dynamic_axes_map = None
    if dynamic_axes:
        dynamic_axes_map = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_map,
        dynamo=False
    )

    # --- PROCEDURA DE UNIFICARE AUTOMATĂ ---
    print("\n[Unificare] PyTorch a terminat exportul. Începem unificarea fișierelor...")
    try:
        # Încărcăm modelul, absorbând automat fișierul extern (.data)
        onnx_model = onnx.load(output_path, load_external_data=True)

        # Resalvăm modelul, forțând scrierea datelor în interiorul acestuia
        onnx.save_model(onnx_model, output_path)

        # Facem curățenie pe disc ștergând fișierul cu extensia .data
        external_data_file = output_path + ".data"
        if os.path.exists(external_data_file):
            os.remove(external_data_file)
            print(f"[Unificare] Am curățat fișierul rezidual: {external_data_file}")

        print(f"✅ SUCCES! Modelul (aprox. 250MB) a fost unificat și salvat perfect în: {output_path}\n")
    except Exception as e:
        print(f"❌ A apărut o eroare la încercarea de unificare: {e}")


def main():
    config = R3DTransferConfig(dataset_name="Mix")
    default_ckpt = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    default_out = config.SAVE_DIR / f"{config.MODEL_NAME}.onnx"

    parser = argparse.ArgumentParser(
        description="Export R3D-18 best checkpoint to ONNX format."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help="Path to .pth checkpoint (default: best checkpoint).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help="Path to write .onnx file.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Enable dynamic batch axis in exported graph.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict state_dict loading.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=112,
        help="Spatial input size used during export.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_onnx(
        checkpoint_path=checkpoint_path,
        output_path=str(output_path),
        device=torch.device(args.device),
        opset=args.opset,
        dynamic_axes=args.dynamic_axes,
        strict=not args.no_strict,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()