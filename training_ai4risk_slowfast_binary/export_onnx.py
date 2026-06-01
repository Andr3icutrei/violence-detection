import argparse
import os  # <-- Adăugat
from pathlib import Path

import onnx  # <-- Adăugat
import torch
import torch.nn as nn

from config import SlowFastConfig
from model import SlowFastViolence


# --- [MODIFICARE NOUĂ] ---
# Wrapper pentru a ascunde parametrul 'return_cam' (sau orice alt argument boolean) de trasorul ONNX.
class SlowFastONNXWrapper(nn.Module):
    def __init__(self, original_model, mean, std):
        super().__init__()
        self.model = original_model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, slow_input, fast_input):
        # Modelele SlowFast din PyTorchVideo se așteaptă ca intrarea să fie
        # o listă de tensori [slow, fast], nu două argumente separate.
        slow_input = (slow_input - self.mean) / self.std
        fast_input = (fast_input - self.mean) / self.std
        inputs = [slow_input, fast_input]

        # Acum trimitem lista ca un singur argument și forțăm return_cam=False
        return self.model(inputs, return_cam=False)


class SlowFastSingleInputWrapper(nn.Module):
    def __init__(self, original_model, slow_indices, mean, std):
        super().__init__()
        self.model = original_model
        self.register_buffer("slow_indices", slow_indices)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, fast_input):
        # Normalizăm înainte de selecția pathway-ului lent.
        fast_input = (fast_input - self.mean) / self.std
        slow_input = fast_input.index_select(dim=2, index=self.slow_indices)
        inputs = [slow_input, fast_input]
        return self.model(inputs, return_cam=False)


def _load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def export_onnx(checkpoint_path, output_path, device, opset, dynamic_axes, strict, single_input):
    config = SlowFastConfig()
    model = SlowFastViolence(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout_p=config.DROPOUT_P,
        slowfast_alpha=config.SLOWFAST_ALPHA,
        slowfast_beta=config.SLOWFAST_BETA,
    ).to(device)

    state_dict = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    mean = torch.tensor(config.KINETICS_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.KINETICS_STD, device=device).view(1, 3, 1, 1, 1)

    if single_input:
        slow_indices = torch.arange(0, config.FAST_FRAMES, config.SLOWFAST_ALPHA)
        slow_indices = slow_indices[:config.SLOW_FRAMES].to(device)
        wrapped_model = SlowFastSingleInputWrapper(model, slow_indices, mean, std)
    else:
        wrapped_model = SlowFastONNXWrapper(model, mean, std)

    wrapped_model.eval()

    if single_input:
        fast_input = torch.randn(
            1,
            3,
            config.FAST_FRAMES,
            config.INPUT_SIZE,
            config.INPUT_SIZE,
            device=device,
        )
        export_inputs = (fast_input,)
        input_names = ["video_input"]
    else:
        slow_input = torch.randn(
            1,
            3,
            config.SLOW_FRAMES,
            config.INPUT_SIZE,
            config.INPUT_SIZE,
            device=device,
        )
        fast_input = torch.randn(
            1,
            3,
            config.FAST_FRAMES,
            config.INPUT_SIZE,
            config.INPUT_SIZE,
            device=device,
        )
        export_inputs = (slow_input, fast_input)
        input_names = ["slow_input", "fast_input"]

    output_names = ["logits"]

    dynamic_axes_map = None
    if dynamic_axes:
        dynamic_axes_map = {
            input_names[0]: {0: "batch"},
            "logits": {0: "batch"},
        }
        if not single_input:
            dynamic_axes_map[input_names[1]] = {0: "batch"}

    torch.onnx.export(
        wrapped_model,
        export_inputs,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_map,
        dynamo=False,
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

        print(f"✅ SUCCES! Modelul SlowFast a fost unificat și salvat perfect în: {output_path}\n")
    except Exception as e:
        print(f"❌ A apărut o eroare la încercarea de unificare: {e}")


def main():
    config = SlowFastConfig()
    default_ckpt = config.SAVE_DIR / f"{config.MODEL_NAME}_best.pth"
    default_out = config.SAVE_DIR / f"{config.MODEL_NAME}.onnx"

    parser = argparse.ArgumentParser(
        description="Export SlowFast best checkpoint to ONNX format."
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
        default=17,  # <-- Modificat la 18 conform warning-urilor tale anterioare
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
        "--single-input",
        action="store_true",
        help="Export ONNX with a single 32-frame input and internal slow/fast split.",
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
        single_input=args.single_input,
    )


if __name__ == "__main__":
    main()