import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import os

# Oprirea logurilor ffmpeg inutile
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'


class X3DVideoDataset(Dataset):
    def __init__(self, violence_path, non_violence_path,
                 num_frames=16, temporal_stride=3,
                 split_ratio=0.8, training=True, augment=True,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225],
                 crop_size=224, seed=42, use_crop=False,
                 use_optical_flow=False):

        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.split_ratio = split_ratio
        self.training = training
        # Augmentarea este activa doar pe setul de antrenare
        self.augment = augment and training
        self.crop_size = crop_size
        self.seed = seed
        self.use_crop = use_crop
        self.use_optical_flow = use_optical_flow

        # Tensorii de normalizare (Kinetics stats)
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        if isinstance(violence_path, dict) and violence_path.get('type') == 'multiclass':
            self.dataset_type = 'multiclass'
            self.base_path = violence_path['path']
            self.violence_dirs = violence_path['violence_dirs']
            self.non_violence_dirs = violence_path['non_violence_dirs']
        else:
            raise ValueError("Only AI4RiSK multiclass dataset is supported")

        # Incarcare cai fisiere
        self.video_paths, self.labels = self._load_video_paths()

    def _load_video_paths(self):
        violent_videos = []
        non_violent_videos = []

        # Setam seed pentru reproductibilitatea impartirii Train/Val
        random.seed(self.seed)
        base_path = Path(self.base_path)

        # 1. Incarcare videouri Violente
        for dir_name in self.violence_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                # Amestecam inainte de split pentru a nu lua doar primele/ultimele
                random.shuffle(dataset_videos)

                split_idx = int(len(dataset_videos) * self.split_ratio)
                if self.training:
                    violent_videos.extend(dataset_videos[:split_idx])
                else:
                    violent_videos.extend(dataset_videos[split_idx:])

        # 2. Incarcare videouri Non-Violente
        for dir_name in self.non_violence_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                dataset_videos = sorted([f for f in dir_path.rglob('*') if f.is_file()])
                random.shuffle(dataset_videos)

                split_idx = int(len(dataset_videos) * self.split_ratio)
                if self.training:
                    non_violent_videos.extend(dataset_videos[:split_idx])
                else:
                    non_violent_videos.extend(dataset_videos[split_idx:])

        # Combinam listele
        videos = violent_videos + non_violent_videos
        labels = [1] * len(violent_videos) + [0] * len(non_violent_videos)

        # Shuffle final pentru antrenare
        combined = list(zip(videos, labels))
        random.shuffle(combined)
        videos, labels = zip(*combined) if combined else ([], [])

        # Reset seed pentru a nu afecta augmentarile ulterioare
        random.seed()

        return list(videos), list(labels)

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convertim BGR -> RGB imediat
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames_consistent(self, frames):
        """
        Extrage frame-urile necesare gestionand corect videourile scurte.
        Evita 'teleportarea' (looping) si foloseste edge padding.
        """
        total_frames = len(frames)
        if total_frames == 0: return None

        # Lungimea totala necesara in frame-uri originale (ex: 16 * 3 = 48)
        seq_len = self.num_frames * self.temporal_stride

        if total_frames < seq_len:
            # FIX: Edge Padding (repetam ultimul frame) in loc de modulo loop
            # Luam tot ce avem
            indices = list(range(total_frames))
            # Completam pana la seq_len repetand ultimul index
            last_idx = total_frames - 1
            while len(indices) < seq_len:
                indices.append(last_idx)
        else:
            # Daca avem destule frame-uri, alegem o fereastra random sau centrata
            if self.training:
                # Random start pentru antrenare (Temporal Jittering)
                start_idx = random.randint(0, total_frames - seq_len)
            else:
                # Centrat pentru validare
                start_idx = (total_frames - seq_len) // 2

            indices = list(range(start_idx, start_idx + seq_len))

        # Aplicam pasul (stride)
        indices = indices[::self.temporal_stride]

        # Safety check: ne asiguram ca avem exact num_frames
        indices = indices[:self.num_frames]

        # Daca totusi nu sunt destule (caz extrem), mai facem padding la final
        while len(indices) < self.num_frames:
            indices.append(indices[-1])

        return [frames[i] for i in indices]

    def _transform_video_consistent(self, frames):
        """
        Aplica augmentari CONSISTENTE temporal.
        Aceiasi parametri (flip, brightness) se aplica pe TOATE frame-urile din secventa.
        """

        # 1. Decidem parametrii de augmentare O SINGURA DATA per video
        do_flip = False
        brightness_factor = 1.0
        contrast_factor = 1.0

        # Parametrii pentru Random Crop
        do_random_crop = False
        crop_top, crop_left = 0, 0

        h_img, w_img = frames[0].shape[:2]

        if self.augment:
            # 50% sanse de Horizontal Flip
            if random.random() > 0.5:
                do_flip = True

            # Color Jitter
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)

            # Random Crop Logic
            if self.use_crop and (h_img > self.crop_size and w_img > self.crop_size):
                do_random_crop = True
                crop_top = random.randint(0, h_img - self.crop_size)
                crop_left = random.randint(0, w_img - self.crop_size)

        processed_frames = []

        for frame in frames:
            frame = frame.astype(np.float32) / 255.0

            # --- Aplicare Augmentari ---

            # 1. Flip
            if do_flip:
                frame = np.fliplr(frame).copy()

            # 2. Color/Brightness
            if brightness_factor != 1.0 or contrast_factor != 1.0:
                # Brightness
                frame = frame * brightness_factor
                # Contrast simple approximation
                mean_val = frame.mean()
                frame = (frame - mean_val) * contrast_factor + mean_val
                frame = np.clip(frame, 0, 1)

            # 3. Crop & Resize
            if do_random_crop:
                frame = frame[crop_top:crop_top + self.crop_size, crop_left:crop_left + self.crop_size]
            else:
                # Daca nu facem random crop sau imaginea e deja mica, facem resize standard
                # Center crop sau simplu resize
                if self.use_crop:
                    # Center crop fallback
                    start_y = (h_img - self.crop_size) // 2
                    start_x = (w_img - self.crop_size) // 2
                    # Asiguram limite pozitive
                    start_y = max(0, start_y)
                    start_x = max(0, start_x)
                    end_y = min(h_img, start_y + self.crop_size)
                    end_x = min(w_img, start_x + self.crop_size)

                    frame = frame[start_y:end_y, start_x:end_x]

                    # Daca dupa crop e inca prea mic, resize up
                    if frame.shape[0] != self.crop_size or frame.shape[1] != self.crop_size:
                        frame = cv2.resize(frame, (self.crop_size, self.crop_size))
                else:
                    # Resize direct (Standard pentru X3D daca nu folosim spatial crops complexe)
                    frame = cv2.resize(frame, (self.crop_size, self.crop_size))

            processed_frames.append(frame)

        return np.stack(processed_frames, axis=0)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]

            frames = self._extract_frames(video_path)

            # Pas 1: Selectie temporala consistenta
            selected_frames = self._sample_frames_consistent(frames)

            # Daca video-ul e corupt sau gol, incercam urmatorul
            if selected_frames is None or len(selected_frames) != self.num_frames:
                new_idx = (idx + 1) % len(self)
                return self.__getitem__(new_idx)

            # Pas 2: Transformari spatiale consistente
            sequence = self._transform_video_consistent(selected_frames)

            # Convert to Tensor: (T, H, W, C) -> (C, T, H, W)
            tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)

            # Normalizare cu media si deviatia standard Kinetics
            tensor = (tensor - self.mean) / self.std

            label = torch.LongTensor([label])[0]

            return tensor, label

        except Exception as e:
            # Fallback robust in caz de eroare la citire
            # print(f"Error loading video {idx}: {e}")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)