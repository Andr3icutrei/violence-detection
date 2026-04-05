import os
import random
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.hub

import numpy as np
import torchvision.transforms.functional as TF

import decord
from tqdm import tqdm

decord.bridge.set_bridge('torch')

CONFIG = {
    "dataset_root": "../../Datasets/RLVS",
    "checkpoint_dir": "./checkpoints",
    "log_dir": "./runs/slowfast_rlvs",
    "alpha": 4,
    "slow_frames": 8,
    "fast_frames": 32,
    "crop_size": 224,
    "epochs": 50,
    "batch_size": 8,
    "num_workers": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "lr_milestones": [20, 35],
    "lr_gamma": 0.1,
    "dropout": 0.5,
    "label_smoothing": 0.1,
    "seed": 42,
    "val_split": 0.2,
    "pretrained": True,
    "mixed_precision": True,
    "clip_grad_norm": 1.0,
    "save_every": 5,
    "early_stopping_patience": 10,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class RLVSDataset(Dataset):
    def __init__(
            self,
            root: str,
            file_list: list,
            slow_frames: int = 8,
            fast_frames: int = 32,
            transform=None,
            is_train: bool = True,
    ):
        self.root = Path(root)
        self.file_list = file_list
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.file_list)

    def _sample_indices(self, total_frames: int, num_frames: int) -> list:
        if total_frames <= num_frames:
            indices = list(range(total_frames))
            pad = num_frames - total_frames
            indices += [total_frames - 1] * pad
            return indices

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return indices.tolist()

    def __getitem__(self, idx):
        video_path, label = self.file_list[idx]

        try:
            # Inițializăm VideoReader strict pe CPU, cu 1 singur thread pentru a evita
            # conflictele cu workerii din DataLoader
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            total_frames = len(vr)

            if total_frames <= 0:
                raise ValueError(f"Total frames <= 0 for {video_path}")

            slow_indices = self._sample_indices(total_frames, self.slow_frames)
            fast_indices = self._sample_indices(total_frames, self.fast_frames)

            # vr.get_batch returnează direct un tensor PyTorch de forma [T, H, W, C]
            slow_raw = vr.get_batch(slow_indices)
            fast_raw = vr.get_batch(fast_indices)

            # Permutăm în [C, T, H, W] așteptat de PyTorch și normalizăm la [0, 1]
            slow_tensor = slow_raw.permute(3, 0, 1, 2).float() / 255.0
            fast_tensor = fast_raw.permute(3, 0, 1, 2).float() / 255.0

        except Exception as e:
            logger.warning(f"Eroare la citirea {video_path}: {e}")
            # În caz de eroare, returnăm un alt sample random
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        if self.transform is not None:
            slow_tensor = self.transform(slow_tensor)
            fast_tensor = self.transform(fast_tensor)

        return {
            "slow": slow_tensor,
            "fast": fast_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": video_path,
        }

def build_mean_std():
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return mean, std

class VideoTransform:
    def __init__(self, crop_size: int, is_train: bool):
        self.crop_size = crop_size
        self.is_train = is_train
        self.mean, self.std = build_mean_std()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, t, h, w = x.shape
        target_short = 256 if self.is_train else self.crop_size

        if h < w:
            new_h = target_short
            new_w = int(w * target_short / h)
        else:
            new_w = target_short
            new_h = int(h * target_short / w)

        frames = []
        for i in range(t):
            frame = x[:, i, :, :]
            frame = TF.resize(frame, [new_h, new_w], antialias=True)
            frames.append(frame)
        x = torch.stack(frames, dim=1)

        _, _, h, w = x.shape
        if self.is_train:
            top  = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
        else:
            top  = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2

        x = x[:, :, top:top + self.crop_size, left:left + self.crop_size]

        if self.is_train and random.random() < 0.5:
            x = torch.flip(x, dims=[-1])

        self.mean = self.mean.to(x.device)
        self.std  = self.std.to(x.device)
        x = (x - self.mean) / self.std

        return x

def build_model(num_classes: int = 2, dropout: float = 0.5, pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="slowfast_r50",
            pretrained=True,
        )
    else:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="slowfast_r50",
            pretrained=False,
        )

    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model

def collect_videos(dataset_root: str) -> list:
    root = Path(dataset_root)
    class_map = {"Violence": 1, "NonViolence": 0}
    samples = []

    for class_name, label in class_map.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Directorul nu exista: {class_dir}")

        videos = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        samples.extend([(str(p), label) for p in videos])

    return samples

def train_val_split(samples: list, val_split: float, seed: int):
    random.seed(seed)

    violence     = [(p, l) for p, l in samples if l == 1]
    non_violence = [(p, l) for p, l in samples if l == 0]

    def split(lst):
        random.shuffle(lst)
        cut = int(len(lst) * val_split)
        return lst[cut:], lst[:cut]

    v_train, v_val = split(violence)
    nv_train, nv_val = split(non_violence)

    train_set = v_train + nv_train
    val_set   = v_val   + nv_val

    random.shuffle(train_set)
    random.shuffle(val_set)

    return train_set, val_set

def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    labels = [l for _, l in samples]
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts
    sample_weights = [weights_per_class[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),
        replacement=True,
    )

def slowfast_collate(batch):
    slow   = torch.stack([item["slow"]  for item in batch])
    fast   = torch.stack([item["fast"]  for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return [slow, fast], labels

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, epoch, writer
):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)

    for step, (inputs, labels) in enumerate(pbar):
        inputs = [x.to(device, non_blocking=True) for x in inputs]
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(inputs)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad_norm"])
            optimizer.step()

        acc = accuracy(logits.detach(), labels)
        bs  = labels.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        global_step = (epoch - 1) * len(loader) + step
        writer.add_scalar("Train/StepLoss", loss.item(), global_step)

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [VAL]", leave=False)

    for inputs, labels in pbar:
        inputs = [x.to(device, non_blocking=True) for x in inputs]
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss   = criterion(logits, labels)

        acc = accuracy(logits, labels)
        bs  = labels.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return loss_meter.avg, acc_meter.avg, precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",   default=CONFIG["dataset_root"])
    parser.add_argument("--epochs",         type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch_size",     type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",             type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--no_pretrained",  action="store_true")
    parser.add_argument("--no_amp",         action="store_true")
    parser.add_argument("--resume",         type=str,   default=None)
    args = parser.parse_args()

    CONFIG["dataset_root"]    = args.dataset_root
    CONFIG["epochs"]          = args.epochs
    CONFIG["batch_size"]      = args.batch_size
    CONFIG["learning_rate"]   = args.lr
    CONFIG["pretrained"]      = not args.no_pretrained
    CONFIG["mixed_precision"] = not args.no_amp

    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(CONFIG["log_dir"])

    all_samples = collect_videos(CONFIG["dataset_root"])

    train_samples, val_samples = train_val_split(
        all_samples, CONFIG["val_split"], CONFIG["seed"]
    )

    train_transform = VideoTransform(crop_size=CONFIG["crop_size"], is_train=True)
    val_transform   = VideoTransform(crop_size=CONFIG["crop_size"], is_train=False)

    train_dataset = RLVSDataset(
        root=CONFIG["dataset_root"],
        file_list=train_samples,
        slow_frames=CONFIG["slow_frames"],
        fast_frames=CONFIG["fast_frames"],
        transform=train_transform,
        is_train=True,
    )
    val_dataset = RLVSDataset(
        root=CONFIG["dataset_root"],
        file_list=val_samples,
        slow_frames=CONFIG["slow_frames"],
        fast_frames=CONFIG["fast_frames"],
        transform=val_transform,
        is_train=False,
    )

    sampler = make_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        collate_fn=slowfast_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=slowfast_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = build_model(
        num_classes=2,
        dropout=CONFIG["dropout"],
        pretrained=CONFIG["pretrained"],
    )
    model = model.to(device)

    backbone_params = [p for n, p in model.named_parameters() if "proj" not in n]
    head_params     = [p for n, p in model.named_parameters() if "proj"     in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": CONFIG["learning_rate"] * 0.1},
        {"params": head_params,     "lr": CONFIG["learning_rate"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=CONFIG["lr_milestones"],
        gamma=CONFIG["lr_gamma"],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    scaler = torch.amp.GradScaler('cuda') if (CONFIG["mixed_precision"] and device.type == "cuda") else None

    start_epoch    = 1
    best_val_acc   = 0.0
    patience_count = 0

    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch  = ckpt["epoch"] + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, writer
        )

        val_loss, val_acc, precision, recall, f1 = validate(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step()

        writer.add_scalars("Loss",      {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy",  {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("Val/F1",        f1,        epoch)
        writer.add_scalar("Val/Precision", precision, epoch)
        writer.add_scalar("Val/Recall",    recall,    epoch)
        writer.add_scalar("LR/backbone",   optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/head",       optimizer.param_groups[1]["lr"], epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc   = val_acc
            patience_count = 0
            best_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "val_acc":      val_acc,
                "best_val_acc": best_val_acc,
                "config":       CONFIG,
            }, best_path)
        else:
            patience_count += 1

        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "val_acc":      val_acc,
                "best_val_acc": best_val_acc,
                "config":       CONFIG,
            }, ckpt_path)

        if patience_count >= CONFIG["early_stopping_patience"]:
            break

    writer.close()

if __name__ == "__main__":
    main()