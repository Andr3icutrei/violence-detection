import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from model import X3DViolence
from config import X3DConfig


class MultiViewFlowDataset:
    def __init__(self, dataset_info, num_frames=16, temporal_stride=2,
                 split_ratio=0.8, training=False, num_clips=10,
                 mean=[0.5, 0.5], std=[0.225, 0.225]):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.split_ratio = split_ratio
        self.training = training
        self.num_clips = num_clips

        self.mean = torch.tensor(mean).view(len(mean), 1, 1, 1)
        self.std = torch.tensor(std).view(len(std), 1, 1, 1)

        self.base_path = Path(dataset_info['path'])
        self.violence_dirs = dataset_info['violence_dirs']
        self.non_violence_dirs = dataset_info['non_violence_dirs']
        self.extension = dataset_info.get('extension', '.npy')

        self.file_paths, self.labels = self._load_file_paths()

    def _load_file_paths(self):
        violent_files, non_violent_files = [], []

        for dir_name in self.violence_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = sorted([f for f in dir_path.rglob(f'*{self.extension}')])
                split_idx = int(len(files) * self.split_ratio)
                if self.training:
                    violent_files.extend(files[:split_idx])
                else:
                    violent_files.extend(files[split_idx:])

        for dir_name in self.non_violence_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = sorted([f for f in dir_path.rglob(f'*{self.extension}')])
                split_idx = int(len(files) * self.split_ratio)
                if self.training:
                    non_violent_files.extend(files[:split_idx])
                else:
                    non_violent_files.extend(files[split_idx:])

        files = violent_files + non_violent_files
        labels = [1] * len(violent_files) + [0] * len(non_violent_files)
        return files, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        full_flow = np.load(str(file_path)).astype(np.float32)
        total_frames = full_flow.shape[0]
        window_size = self.num_frames * self.temporal_stride

        processed_clips = []
        if total_frames <= window_size:
            start_indices = [0]
        else:
            step = max(1, (total_frames - window_size) // (self.num_clips - 1))
            start_indices = [min(i * step, total_frames - window_size) for i in range(self.num_clips)]

        for start in start_indices:
            clip = full_flow[start: start + window_size]
            if clip.shape[0] < window_size:
                pad_len = window_size - clip.shape[0]
                padding = np.tile(clip[-1:], (pad_len, 1, 1, 1))
                clip = np.concatenate([clip, padding], axis=0)

            clip = clip[::self.temporal_stride][:self.num_frames]
            tensor = torch.from_numpy(clip).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std
            processed_clips.append(tensor)

        return processed_clips, torch.LongTensor([label])[0]


def evaluate_model_multiview(model_path, config, num_clips=10):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    model = X3DViolence(
        num_classes=2, pretrained=False,
        x3d_version=config.X3D_VERSION, input_channels=config.INPUT_CHANNELS
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_dataset = MultiViewFlowDataset(
        dataset_info=config.VIOLENCE_PATH,
        num_frames=config.NUM_FRAMES, temporal_stride=config.TEMPORAL_STRIDE,
        split_ratio=config.SPLIT_RATIO, training=False, num_clips=num_clips,
        mean=config.KINETICS_MEAN, std=config.KINETICS_STD
    )

    if len(val_dataset) == 0:
        return

    correct, total = 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for clips, label in tqdm(val_dataset, desc="Evaluating"):
            label = label.to(device)
            clip_outputs = []
            for clip in clips:
                output = model(clip.unsqueeze(0).to(device))
                clip_outputs.append(output)

            max_output, _ = torch.max(torch.stack(clip_outputs), dim=0)
            probs = torch.softmax(max_output, dim=1)
            _, predicted = torch.max(max_output, 1)

            total += 1
            correct += (predicted == label).sum().item()
            all_preds.append(predicted.cpu().numpy()[0])
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs[0, 1].cpu().numpy())

    accuracy = 100 * correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    print(f"\nAccuracy: {accuracy:.2f}%, AUC: {auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence']))