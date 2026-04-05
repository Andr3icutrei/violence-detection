import os
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Importăm clasele și funcțiile necesare din scriptul tău de antrenare
# Asigură-te că fișierul tău principal de antrenare se numește train.py
from train import (
    RLVSDataset,
    VideoTransform,
    build_model,
    slowfast_collate,
    collect_videos,
    train_val_split,
    CONFIG
)


class SlowFastGradCAM:
    """
    Implementare custom Grad-CAM adaptată pentru arhitectura SlowFast folosind Tensor Hooks.
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Ne legăm la ultimul bloc ResNet (blocks[4]). blocks[5] este head-ul de clasificare.
        self.target_layer = self.model.blocks[4]
        self.target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        # output este lista: [slow_features, fast_features]
        fast_features = output[1]
        self.activations = fast_features.detach()

        # Punem hook DIRECT pe tensor pentru a extrage gradienții,
        # evitând limitarea PyTorch legată de modulele care returnează liste.
        fast_features.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate_heatmap(self, inputs, target_class=None):
        self.model.zero_grad()

        logits = self.model(inputs)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        loss = logits[0, target_class]
        loss.backward()

        if self.gradients is None:
            raise ValueError("Gradienții sunt None. Asigură-te că input-urile au requires_grad=True.")

        # self.gradients are forma: [Batch, Channels, Time, Height, Width]
        # Global Average Pooling pe dimensiunile spațio-temporale (păstrăm dimensiunea pentru broadcast)
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)

        # Extragem primul (și singurul) element din batch
        activations = self.activations[0]  # [Channels, Time, Height, Width]
        pooled_gradients = pooled_gradients[0]  # [Channels, 1, 1, 1]

        # Ponderăm activările prin broadcasting
        weighted_activations = activations * pooled_gradients

        # Facem media pe canal (dim=0)
        heatmap = torch.mean(weighted_activations, dim=0)  # [Time, Height, Width]

        # Aplicăm ReLU pentru a păstra doar influențele pozitive asupra clasei
        heatmap = F.relu(heatmap)

        # Normalizăm între 0 și 1 în siguranță
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap.cpu().numpy(), target_class, logits[0].detach().cpu().numpy()


def save_gradcam_video(video_path, heatmaps, output_path, transform, original_frames_count=32):
    """
    Suprapune heatmap-ul Grad-CAM peste cadrele originale și salvează un video.
    """
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))

    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, original_frames_count, dtype=int).tolist()
    frames = vr.get_batch(indices).numpy()  # [T, H, W, C]

    T, H, W, C = frames.shape

    # Setăm writer-ul pentru video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (W, H))

    # Heatmaps are dimensiunea [Time, Map_H, Map_W]
    # Trebuie să facem resize la fiecare heatmap pentru a se potrivi cu HxW
    for i in range(min(T, heatmaps.shape[0])):
        frame = frames[i]  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hm = heatmaps[i]
        hm = cv2.resize(hm, (W, H))
        hm = np.uint8(255 * hm)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # Suprapunem (0.6 cadru original, 0.4 heatmap)
        overlay = cv2.addWeighted(frame_bgr, 0.6, hm_color, 0.4, 0)
        out.write(overlay)

    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Calea catre best_model.pth")
    parser.add_argument("--dataset_root", type=str, default=CONFIG["dataset_root"])
    parser.add_argument("--out_dir", type=str, default="./inference_output")
    parser.add_argument("--num_gradcam_samples", type=int, default=5,
                        help="Numarul de video-uri pentru care generam heatmap")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Incarcare model...")
    model = build_model(num_classes=2, dropout=CONFIG["dropout"], pretrained=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print("[INFO] Incarcare date de validare...")
    all_samples = collect_videos(args.dataset_root)
    _, val_samples = train_val_split(all_samples, CONFIG["val_split"], CONFIG["seed"])

    val_transform = VideoTransform(crop_size=CONFIG["crop_size"], is_train=False)
    val_dataset = RLVSDataset(
        root=args.dataset_root,
        file_list=val_samples,
        slow_frames=CONFIG["slow_frames"],
        fast_frames=CONFIG["fast_frames"],
        transform=val_transform,
        is_train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=slowfast_collate,
    )

    # 1. EVALUARE METRICI
    print("[INFO] Evaluare pe setul de validare...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Testing"):
            inputs = [x.to(device) for x in inputs]
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Afișare Metrici
    print("\n" + "=" * 50)
    print("RAPORT DE CLASIFICARE:")
    print(classification_report(all_labels, all_preds, target_names=["NonViolence", "Violence"]))
    print("=" * 50)

    cm = confusion_matrix(all_labels, all_preds)
    print("MATRICE DE CONFUZIE:")
    print(cm)
    print("=" * 50)

    # 2. GRAD-CAM
    print(f"\n[INFO] Generare Grad-CAM pentru {args.num_gradcam_samples} sample-uri random...")
    gradcam = SlowFastGradCAM(model)

    # Selectăm sample-uri random din dataset-ul de validare
    sample_indices = random.sample(range(len(val_dataset)), args.num_gradcam_samples)

    for idx in tqdm(sample_indices, desc="Grad-CAM"):
        item = val_dataset[idx]

        # Adăugăm .requires_grad_(True) pentru a forța calculul gradienților necesari pentru Grad-CAM
        slow = item["slow"].unsqueeze(0).to(device).requires_grad_(True)
        fast = item["fast"].unsqueeze(0).to(device).requires_grad_(True)

        label = item["label"].item()
        v_path = item["path"]

        inputs = [slow, fast]

        # Generăm heatmap (va calcula direct backward() in interior)
        heatmaps, pred_class, logits = gradcam.generate_heatmap(inputs)

        # Salvăm video-ul rezultat
        filename = os.path.basename(v_path)
        out_video_path = os.path.join(args.out_dir, f"cam_true{label}_pred{pred_class}_{filename}")

        save_gradcam_video(
            video_path=v_path,
            heatmaps=heatmaps,
            output_path=out_video_path,
            transform=val_transform,
            original_frames_count=CONFIG["fast_frames"]
        )

    print(f"[INFO] Gata! Rezultatele Grad-CAM au fost salvate in: {args.out_dir}")


if __name__ == "__main__":
    main()