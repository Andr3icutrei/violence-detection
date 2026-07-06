from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import json
from collections import Counter
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
from model import MViTViolence
from dataset import MViTVideoDataset
from config import MViTConfig


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        """Initialize the object and its runtime state."""
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        """Update the early-stopping state with the latest validation loss."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def _compute_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of all available parameter gradients."""
    total_norm: float = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def _print_epoch_metrics(split_name: str, loss: float, all_labels: List[int], all_preds: List[int], class_names: List[str]) -> None:
    """Print compact metrics and the confusion matrix for one epoch split."""
    acc: float = accuracy_score(all_labels, all_preds)
    f1_macro: float = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class: np.ndarray = f1_score(all_labels, all_preds, average=None, zero_division=0)
    recall_macro: float = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_per_class: np.ndarray = recall_score(all_labels, all_preds, average=None, zero_division=0)
    cm: np.ndarray = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    print(f'\n{split_name} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | F1 macro: {f1_macro:.4f} | Recall macro: {recall_macro:.4f}')
    header: str = f"{'Class':<22} {'F1':>8} {'Recall':>8}"
    print(header)
    print('-' * len(header))
    for i, name in enumerate(class_names):
        f1_val: float = f1_per_class[i] if i < len(f1_per_class) else 0.0
        rec_val: float = recall_per_class[i] if i < len(recall_per_class) else 0.0
        print(f'{name:<22} {f1_val:>8.4f} {rec_val:>8.4f}')
    print('\nConfusion Matrix:')
    col_width: int = 6
    header_row: str = ' ' * 22 + ''.join((f'{n[:col_width]:>{col_width}}' for n in class_names))
    print(header_row)
    for i, name in enumerate(class_names):
        row: str = f'{name:<22}' + ''.join((f'{cm[i, j]:>{col_width}}' for j in range(len(class_names))))
        print(row)
    print()


class MViTTrainer:
    """Coordinate MViT training, validation, optimization, and checkpointing.

    The entire MViT model is trained from the first epoch: there is no frozen
    backbone and no unfreeze schedule.
    """

    def __init__(self, config: MViTConfig) -> None:
        """Initialize the object and its runtime state."""
        self.config: MViTConfig = config
        self.device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model: MViTViolence = MViTViolence(num_classes=config.NUM_CLASSES, pretrained=config.USE_PRETRAINED, dropout_p=config.DROPOUT_P).to(self.device)
        self.train_loader: DataLoader = self._create_dataloader(training=True)
        self.val_loader: DataLoader = self._create_dataloader(training=False)
        class_weights: Optional[torch.Tensor] = None
        if config.USE_CLASS_WEIGHTS:
            weights_list = self._calculate_class_weights()
            class_weights = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self._setup_optimizer()
        self.scaler: Optional[GradScaler] = GradScaler() if config.USE_AMP else None
        self.early_stopping: EarlyStopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        self.history: dict = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'learning_rates': []}

    def _calculate_class_weights(self) -> List[float]:
        """Compute normalized inverse-frequency class weights."""
        all_labels: List[int] = self.train_loader.dataset.labels
        label_counts: Counter = Counter(all_labels)
        total_samples: int = len(all_labels)
        num_classes: int = self.config.NUM_CLASSES
        weights: List[float] = []
        for i in range(num_classes):
            count: int = label_counts.get(i, 1)
            weight: float = total_samples / (num_classes * count)
            weights.append(weight)
        max_weight: float = max(weights)
        weights = [w / max_weight for w in weights]
        return weights

    def _setup_optimizer(self) -> None:
        """Create optimizer parameter groups and the optional scheduler.

        All parameters are trainable. Two groups are used only so the pretrained
        backbone can run at a lower LR than the freshly-initialized head; both
        groups train from epoch 1.
        """
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        param_groups: List[dict] = [
            {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
            {'params': head_params, 'lr': self.config.HEAD_LR},
        ]
        optimizer_type: str = self.config.OPTIMIZER.lower()
        if optimizer_type == 'adamw':
            self.optimizer: optim.Optimizer = optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY, betas=self.config.BETAS, eps=self.config.EPS)
        else:
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.config.WEIGHT_DECAY)
        if self.config.USE_SCHEDULER:
            scheduler_type: str = self.config.SCHEDULER_TYPE
            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.T_0, T_mult=self.config.T_MULT, eta_min=self.config.ETA_MIN)
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            elif scheduler_type == 'reduce_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def _create_dataloader(self, training: bool, batch_size: Optional[int] = None) -> DataLoader:
        """Build an MViT dataloader for training or validation."""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        violence_path: dict = self.config.VIOLENCE_PATH
        non_violence_path: dict = self.config.NON_VIOLENCE_PATH
        dataset = MViTVideoDataset(violence_path=violence_path, non_violence_path=non_violence_path, num_frames=self.config.NUM_FRAMES, temporal_stride=self.config.TEMPORAL_STRIDE, split_ratio=self.config.SPLIT_RATIO, training=training, augment=True, mean=self.config.KINETICS_MEAN, std=self.config.KINETICS_STD, crop_size=self.config.CROP_SIZE, seed=self.config.SEED, use_crop=self.config.USE_CROP)
        sampler = None
        shuffle = training
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=self.config.NUM_WORKERS, pin_memory=self.config.PIN_MEMORY)

    def train_epoch(self, pbar: tqdm) -> Tuple[float, float, List[int], List[int]]:
        """Run one training epoch and return aggregate metrics."""
        self.model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        last_grad_norm: float = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        self.optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if self.config.USE_AMP:
                with autocast(device_type='cuda'):
                    outputs: torch.Tensor = self.model(inputs)
                    loss: torch.Tensor = self.criterion(outputs, labels)
                    loss = loss / self.config.ACCUMULATION_STEPS
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.ACCUMULATION_STEPS
                loss.backward()
            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                if self.config.USE_AMP:
                    self.scaler.unscale_(self.optimizer)
                last_grad_norm = _compute_grad_norm(self.model)
                if self.config.USE_AMP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                    self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item() * self.config.ACCUMULATION_STEPS * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            running_f1: float = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            running_recall: float = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            pbar.set_postfix({'phase': 'train', 'loss': f'{loss.item() * self.config.ACCUMULATION_STEPS:.4f}', 'acc': f'{100 * correct / total:.2f}%', 'f1': f'{running_f1:.4f}', 'recall': f'{running_recall:.4f}', 'grad_norm': f'{last_grad_norm:.2f}'})
            pbar.update(1)
        if (batch_idx + 1) % self.config.ACCUMULATION_STEPS != 0:
            if self.config.USE_AMP:
                self.scaler.unscale_(self.optimizer)
            last_grad_norm = _compute_grad_norm(self.model)
            if self.config.USE_AMP:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.optimizer.step()
            self.optimizer.zero_grad()
        epoch_loss: float = running_loss / total
        epoch_acc: float = correct / total
        _print_epoch_metrics('TRAIN', epoch_loss, all_labels, all_preds, self.config.CLASS_NAMES)
        return (epoch_loss, epoch_acc, all_labels, all_preds)

    def validate_epoch(self, pbar: tqdm) -> Tuple[float, float, List[int], List[int]]:
        """Run one validation epoch and return aggregate metrics."""
        self.model.eval()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        all_preds: List[int] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if self.config.USE_AMP:
                    with autocast(device_type='cuda'):
                        outputs: torch.Tensor = self.model(inputs)
                        loss: torch.Tensor = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                running_f1: float = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                running_recall: float = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                pbar.set_postfix({'phase': 'val', 'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%', 'f1': f'{running_f1:.4f}', 'recall': f'{running_recall:.4f}'})
                pbar.update(1)
        epoch_loss: float = running_loss / total
        epoch_acc: float = correct / total
        _print_epoch_metrics('VAL', epoch_loss, all_labels, all_preds, self.config.CLASS_NAMES)
        return (epoch_loss, epoch_acc, all_labels, all_preds)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save the current training state to disk."""
        checkpoint: dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'history': self.history}
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        filename: str = self.config.SAVE_DIR / f'{self.config.MODEL_NAME}_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        if is_best:
            best_filename: str = self.config.SAVE_DIR / f'{self.config.MODEL_NAME}_best.pth'
            torch.save(checkpoint, best_filename)

    def train(self) -> None:
        """Run the multi-epoch training loop with validation and checkpointing."""
        best_val_loss: float = float('inf')
        trainable: int = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        print(f'Trainable parameters: {trainable:,}')
        for epoch in range(self.config.NUM_EPOCHS):
            print(f'\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}')
            print('-' * 50)
            total_batches: int = len(self.train_loader) + len(self.val_loader)
            pbar: tqdm = tqdm(total=total_batches, desc=f'Epoch {epoch + 1}')
            train_loss, train_acc, train_labels, train_preds = self.train_epoch(pbar)
            val_loss, val_acc, val_labels, val_preds = self.validate_epoch(pbar)
            pbar.close()
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            current_lrs: List[float] = [group['lr'] for group in self.optimizer.param_groups]
            self.history['learning_rates'].append(current_lrs)
            train_f1: float = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            val_f1: float = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            if len(current_lrs) > 1:
                print(f'Learning Rates: Backbone={current_lrs[0]:.2e}, Head={current_lrs[-1]:.2e}')
            else:
                print(f'Learning Rate: {current_lrs[0]:.2e}')
            is_best: bool = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                break
        history_path: str = self.config.SAVE_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f'\nTraining history saved to {history_path}')


# Backward-compatible alias so modules importing the old name keep working.
SlowFastTrainer = MViTTrainer


def main() -> None:
    """Parse command-line arguments and run the selected pipeline mode."""
    config: MViTConfig = MViTConfig()
    print(f'Device: {config.DEVICE}')
    print(f'Training MViT-B (16x4) on {config.DATASET_NAME} dataset')
    print(f'Pretrained: {config.USE_PRETRAINED}')
    print(f'Num frames: {config.NUM_FRAMES}, Temporal stride: {config.TEMPORAL_STRIDE}')
    print(f'Dropout: {config.DROPOUT_P}')
    print(f'Label Smoothing: {config.LABEL_SMOOTHING}')
    print(f'Backbone LR: {config.BACKBONE_LR}, Head LR: {config.HEAD_LR}')
    if config.USE_SCHEDULER:
        print(f'Scheduler: {config.SCHEDULER_TYPE}')
    if config.USE_AMP:
        print('AMP: Enabled')
    trainer: MViTTrainer = MViTTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()