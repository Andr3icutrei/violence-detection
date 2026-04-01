import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from train import MobileNetV2_Attention, create_dataloaders


# ==========================================
# 1. WRAPPER PENTRU EXTRAGEREA ATENȚIEI
# ==========================================
class EvalModelWithAttention(MobileNetV2_Attention):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        self.saved_attn_weights = None

    def forward(self, x):
        b, t, c, h, w = x.size()
        x_view = x.view(b * t, c, h, w)

        features = self.feature_extractor(x_view)
        features = self.pool(features)
        features = features.view(b, t, -1)

        # Extragem atât output-ul cât și greutățile atenției
        attn_output, attn_weights = self.attention(features, features, features)

        # Salvăm greutățile pentru vizualizare (forma: Batch, Target_Seq, Source_Seq)
        self.saved_attn_weights = attn_weights.detach()

        pooled_output = torch.mean(attn_output, dim=1)
        logits = self.classifier(pooled_output)
        return logits


# ==========================================
# 2. IMPLEMENTARE GRAD-CAM (Corectată)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        target = output[0, target_class]
        target.backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3), keepdims=True)  # (T, C, 1, 1)
        cam = np.sum(weights * activations, axis=1)  # (T, H, W)
        cam = np.maximum(cam, 0)  # ReLU

        # CORECTURĂ: Normalizare GLOBALĂ pe toată secvența de 16 cadre
        cams_max = np.max(cam)

        processed_cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            if cams_max != 0:
                c = c / cams_max  # Împărțim la maximul global

            # Resize la rezoluția spațială de intrare (64x64)
            c = cv2.resize(c, (64, 64))
            processed_cams.append(c)

        # Returnăm și greutățile de atenție salvate în model
        attn_w = self.model.saved_attn_weights[0].cpu().numpy()  # (16, 16)
        # Media atenției pe cadre pentru a vedea importanța fiecărui cadru
        frame_importance = np.mean(attn_w, axis=0)

        return np.array(processed_cams), target_class, frame_importance


# ==========================================
# 3. VIZUALIZARE (Rânduri: Original, Grad-CAM, Atenție)
# ==========================================
def save_heatmap_video(frames_tensor, cams, attn_weights, save_path):
    frames = frames_tensor.detach().squeeze(0).permute(0, 2, 3, 1).cpu().numpy()

    # Denormalizare pentru vizualizare clară
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frames = std * frames + mean
    frames = np.clip(frames, 0, 1)
    frames = (frames * 255).astype(np.uint8)

    T = frames.shape[0]

    # Creăm un grid cu 3 rânduri (Original, Grad-CAM, Grafic Atenție)
    fig = plt.figure(figsize=(T * 2, 7))
    gs = fig.add_gridspec(3, T, height_ratios=[1, 1, 0.8])

    # Rândul 1 & 2: Imagini și Heatmaps
    for t in range(T):
        ax1 = fig.add_subplot(gs[0, t])
        ax1.imshow(frames[t])
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1, t])
        heatmap = cv2.applyColorMap(np.uint8(255 * cams[t]), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frames[t], 0.6, heatmap, 0.4, 0)
        ax2.imshow(overlay)
        ax2.axis('off')

    # Rândul 3: Grafic cu distribuția atenției temporale
    ax3 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(T)
    ax3.bar(x_pos, attn_weights, color='coral', edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"Fr {i}" for i in range(T)])
    ax3.set_ylabel('Attention Weight')
    ax3.set_title('Temporal Attention Distribution (Importance of each frame)')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==========================================
# 4. EVALUARE PRINCIPALĂ
# ==========================================
def evaluate_and_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Încarcă loaderele (asigură-te că folosești funcția ta corectă cu num_workers=0 dacă ești pe Windows)
    _, _, test_loader = create_dataloaders(dataset_dir="../../Datasets/RLVS")

    # Folosim modelul cu wrapper pentru atenție
    model = EvalModelWithAttention(num_classes=2).to(device)
    model_path = 'saves/best_mobilenetv2_attention.pth'

    if os.path.exists(model_path):
        # Ignorăm structurile strict egale dacă am adăugat proprietăți noi
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Loaded best model weights.")
    else:
        print(f"Warning: {model_path} not found. Running with untrained weights.")

    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating test set...")
    with torch.no_grad():
        eval_bar = tqdm(test_loader, desc="Test Evaluation")
        correct = 0
        total = 0
        for inputs, labels in eval_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            eval_bar.set_postfix(acc=f"{100 * correct / total:.2f}%")

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Salvare Matrice de Confuzie
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NonViolence", "Violence"],
                yticklabels=["NonViolence", "Violence"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Generare Heatmaps
    print("\nGenerating Grad-CAM Heatmaps and Attention Plots...")
    target_layer = model.feature_extractor[-1]
    grad_cam = GradCAM(model, target_layer)

    heatmap_dir = 'heatmaps'
    os.makedirs(heatmap_dir, exist_ok=True)

    model.eval()
    sample_count = 0
    max_samples = 5

    for inputs, labels in test_loader:
        if sample_count >= max_samples:
            break

        for i in range(inputs.size(0)):
            if sample_count >= max_samples:
                break

            input_tensor = inputs[i].unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            true_label = labels[i].item()

            # Extragem hărțile termice și importanța cadrelor (atenția)
            cams, pred_class, frame_importance = grad_cam.generate(input_tensor, target_class=true_label)

            save_path = os.path.join(heatmap_dir, f'sample_{sample_count}_true_{true_label}_pred_{pred_class}.png')
            save_heatmap_video(input_tensor, cams, frame_importance, save_path)

            sample_count += 1

    print(f"Heatmaps and Attention plots saved in '{heatmap_dir}/' directory.")


if __name__ == '__main__':
    evaluate_and_gradcam()