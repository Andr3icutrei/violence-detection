import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random

class X3DFlowDataset(Dataset):
    def __init__(self, dataset_info, num_frames=16, temporal_stride=2,
                 split_ratio=0.8, training=True, augment=True,
                 mean=[0.5, 0.5], std=[0.225, 0.225], seed=42):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.split_ratio = split_ratio
        self.training = training
        self.augment = augment and training
        self.seed = seed

        self.mean = torch.tensor(mean).view(len(mean), 1, 1, 1)
        self.std = torch.tensor(std).view(len(std), 1, 1, 1)

        self.base_path = Path(dataset_info['path'])
        self.violence_dirs = dataset_info['violence_dirs']
        self.non_violence_dirs = dataset_info['non_violence_dirs']
        self.extension = dataset_info.get('extension', '.npy')

        self.file_paths, self.labels = self._load_file_paths()

    def _load_file_paths(self):
        violent_files = []
        non_violent_files = []

        random.seed(self.seed)

        for dir_name in self.violence_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = sorted([f for f in dir_path.rglob(f'*{self.extension}')])
                random.shuffle(files)
                split_idx = int(len(files) * self.split_ratio)
                if self.training:
                    violent_files.extend(files[:split_idx])
                else:
                    violent_files.extend(files[split_idx:])

        for dir_name in self.non_violence_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = sorted([f for f in dir_path.rglob(f'*{self.extension}')])
                random.shuffle(files)
                split_idx = int(len(files) * self.split_ratio)
                if self.training:
                    non_violent_files.extend(files[:split_idx])
                else:
                    non_violent_files.extend(files[split_idx:])

        all_files = violent_files + non_violent_files
        labels = [1] * len(violent_files) + [0] * len(non_violent_files)

        combined = list(zip(all_files, labels))
        random.shuffle(combined)

        if not combined:
            return [], []

        final_files, final_labels = zip(*combined)
        random.seed()

        return list(final_files), list(final_labels)

    def _process_flow_data(self, flow_data):
        total_frames = flow_data.shape[0]
        needed_frames = self.num_frames * self.temporal_stride

        if total_frames < needed_frames:
            indices = list(range(total_frames))
            while len(indices) < needed_frames:
                indices.extend(indices)
            indices = indices[:needed_frames]
        else:
            if self.training:
                start = random.randint(0, total_frames - needed_frames)
            else:
                start = (total_frames - needed_frames) // 2
            indices = range(start, start + needed_frames)

        clip = flow_data[indices]
        clip = clip[::self.temporal_stride]
        clip = clip[:self.num_frames]
        clip = clip.astype(np.float32)

        if self.augment and random.random() > 0.5:
            clip = np.flip(clip, axis=2).copy()
            clip[:, :, :, 0] = 1.0 - clip[:, :, :, 0]

        return clip

    def __getitem__(self, idx):
        try:
            file_path = self.file_paths[idx]
            label = self.labels[idx]

            flow_data = np.load(str(file_path))
            clip_data = self._process_flow_data(flow_data)

            tensor = torch.from_numpy(clip_data).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std

            return tensor, torch.LongTensor([label])[0]

        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

    def __len__(self):
        return len(self.file_paths)