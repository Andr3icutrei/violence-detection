import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from teacher_model import R3D18Violence
from config import R3DTransferConfig


def visualize_cam_comparison(model_path, video_path, config, output_dir="./visualizations"):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < config.N_FRAMES:
        return

    start_idx = len(frames) // 2 - config.N_FRAMES // 2
    sampled_frames = frames[start_idx:start_idx + config.N_FRAMES]

    original_frame = sampled_frames[config.N_FRAMES // 2]
    original_h, original_w = original_frame.shape[:2]

    processed_frames = []
    for frame in sampled_frames:
        frame_normalized = frame.astype(np.float32) / 255.0
        frame_resized = cv2.resize(frame_normalized, (112, 112))
        processed_frames.append(frame_resized)

    sequence = np.stack(processed_frames, axis=0)
    mean_tensor = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
    std_tensor = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)
    sequence_tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
    sequence_tensor = (sequence_tensor - mean_tensor) / std_tensor

    input_tensor = sequence_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    with torch.enable_grad():
        output = model(input_tensor, return_cam=True)
        pred_class = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, pred_class].backward()

        cam_gradcam = model.get_spatial_cam(pred_class)
        cam_gradcam_pp = model.get_spatial_cam_plus_plus(pred_class)

    heatmap_gradcam = cam_gradcam[0].cpu().numpy()
    heatmap_gradcam = cv2.resize(heatmap_gradcam, (original_w, original_h))

    heatmap_gradcam_pp = cam_gradcam_pp[0].cpu().numpy()
    heatmap_gradcam_pp = cv2.resize(heatmap_gradcam_pp, (original_w, original_h))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original_frame)
    axes[0, 0].set_title('Original Frame', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmap_gradcam, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    overlay_gradcam = original_frame.astype(np.float32) / 255.0
    heatmap_colored = cv2.applyColorMap((heatmap_gradcam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay_gradcam = cv2.addWeighted(overlay_gradcam, 0.6, heatmap_colored, 0.4, 0)
    axes[0, 2].imshow(overlay_gradcam)
    axes[0, 2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(original_frame)
    axes[1, 0].set_title('Original Frame', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmap_gradcam_pp, cmap='jet')
    axes[1, 1].set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    overlay_gradcam_pp = original_frame.astype(np.float32) / 255.0
    heatmap_colored_pp = cv2.applyColorMap((heatmap_gradcam_pp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored_pp = cv2.cvtColor(heatmap_colored_pp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay_gradcam_pp = cv2.addWeighted(overlay_gradcam_pp, 0.6, heatmap_colored_pp, 0.4, 0)
    axes[1, 2].imshow(overlay_gradcam_pp)
    axes[1, 2].set_title('Grad-CAM++ Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    class_names = ['Non-Violent', 'Violent']
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    fig.suptitle(f'Prediction: {class_names[pred_class]} (confidence: {confidence:.2%})\n'
                 f'Video: {Path(video_path).name}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    save_path = output_path / f"{Path(video_path).stem}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved at: {save_path}")


def main():
    config = R3DTransferConfig(dataset_name='Hockey', use_smart_crop=False)
    model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    violence_videos = list(config.VIOLENCE_PATH.rglob('*'))[:3]
    non_violence_videos = list(config.NON_VIOLENCE_PATH.rglob('*'))[:3]

    for video_path in violence_videos + non_violence_videos:
        if video_path.is_file():
            print(f"Processing: {video_path.name}")
            visualize_cam_comparison(model_path, video_path, config)


if __name__ == "__main__":
    main()