import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score,
    confusion_matrix, classification_report
)

from model import X3DViolence
from dataset import X3DVideoDataset
from config import X3DConfig


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return

        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def compute_class_weights(dataset, num_classes, device):
    counts = dataset.get_class_counts()
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def print_metrics(phase, loss, all_labels, all_preds, class_names):
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    target_names = [class_names[i] for i in range(len(class_names))]

    print(f"\n{phase} Metrics:")
    print(f"  Loss:              {loss:.4f}")
    print(f"  Accuracy:          {acc * 100:.2f}%")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  Recall (macro):    {recall_macro:.4f}")
    print(f"\n{phase} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    print(f"{phase} Confusion Matrix:")
    header = f"{'':>20}" + "".join(f"{name:>18}" for name in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:>20}" + "".join(f"{val:>18}" for val in row)
        print(row_str)
    print()

    return {
        'loss': loss,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'recall_macro': recall_macro,
        'confusion_matrix': cm.tolist()
    }


class X3DTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.train_dataset = self._create_dataset(training=True)
        self.val_dataset = self._create_dataset(training=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        self.model = X3DViolence(
            num_classes=config.NUM_CLASSES,
            pretrained=config.USE_PRETRAINED,
            dropout_p=config.DROPOUT_P,
            x3d_version=config.X3D_VERSION
        ).to(self.device)

        class_weights = compute_class_weights(self.train_dataset, config.NUM_CLASSES, self.device)
        print("Class weights:")
        for c in range(config.NUM_CLASSES):
            print(f"  {config.CLASS_NAMES[c]}: {class_weights[c].item():.4f}")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )

        self._setup_optimizer()

        early_stop_mode = 'max' if config.BEST_MODEL_METRIC == 'f1_macro' else 'min'
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode=early_stop_mode
        )

        self.history = {
            'train': [], 'val': [], 'learning_rates': []
        }

    def _create_dataset(self, training):
        return X3DVideoDataset(
            dataset_info=self.config.DATASET_INFO,
            num_frames=self.config.NUM_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            split_ratio=self.config.SPLIT_RATIO,
            split_seed=self.config.SPLIT_SEED,
            training=training,
            augment=True,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP,
            max_retries=self.config.DATASET_MAX_RETRIES,
            aug_flip_prob=self.config.AUG_FLIP_PROB,
            aug_color_prob=self.config.AUG_COLOR_PROB,
            aug_brightness_range=self.config.AUG_BRIGHTNESS_RANGE,
            aug_contrast_range=self.config.AUG_CONTRAST_RANGE,
            aug_rotation_prob=self.config.AUG_ROTATION_PROB,
            aug_rotation_max_degrees=self.config.AUG_ROTATION_MAX_DEGREES,
            aug_cutout_prob=self.config.AUG_CUTOUT_PROB,
            aug_cutout_size_ratio=self.config.AUG_CUTOUT_SIZE_RATIO
        )

    def _setup_optimizer(self):
        # Always include all parameters in the optimizer regardless of freeze state.
        # Freezing is handled via requires_grad; no need to reconstruct the optimizer
        # or scheduler when unfreezing, which would reset scheduler state.
        if self.config.FREEZE_BACKBONE:
            for name, param in self.model.named_parameters():
                if 'blocks.5' not in name and 'proj' not in name:
                    param.requires_grad = False

        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'blocks.5' in name or 'proj' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if self.config.OPTIMIZER.lower() == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ],
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS,
                eps=self.config.EPS
            )
        else:
            self.optimizer = optim.SGD([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ],
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )

        if self.config.USE_SCHEDULER:
            scheduler_type = self.config.SCHEDULER_TYPE

            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.config.T_0,
                    T_mult=self.config.T_MULT,
                    eta_min=self.config.ETA_MIN
                )
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=0.1
                )
            elif scheduler_type == 'reduce_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.1,
                    patience=5
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def _unfreeze_backbone(self):
        # Unfreeze all backbone parameters without touching the optimizer or scheduler.
        # The optimizer already tracks these parameters; enabling their gradients
        # is sufficient for them to receive updates on the next step.
        print("Unfreezing backbone")
        for param in self.model.parameters():
            param.requires_grad = True

    def _run_epoch(self, loader, training):
        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc="Train" if training else "Val")

        if training:
            self.optimizer.zero_grad()

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if training:
                    (loss / self.config.ACCUMULATION_STEPS).backward()

                    if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc * 100:.2f}%'
                })

        if training and (len(loader) % self.config.ACCUMULATION_STEPS != 0):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad()

        epoch_loss = running_loss / max(len(all_labels), 1)
        return epoch_loss, all_labels, all_preds

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            best_filename = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_best.pth"
            torch.save(checkpoint, best_filename)

        if (epoch + 1) % 5 == 0:
            filename = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_epoch_{epoch}.pth"
            torch.save(checkpoint, filename)

    def _is_new_best(self, val_metrics, best_value):
        metric = self.config.BEST_MODEL_METRIC
        if metric == 'f1_macro':
            return val_metrics['f1_macro'] > best_value, val_metrics['f1_macro']
        else:
            return val_metrics['loss'] < best_value, val_metrics['loss']

    def train(self):
        metric = self.config.BEST_MODEL_METRIC
        best_value = -float('inf') if metric == 'f1_macro' else float('inf')
        class_names = self.config.CLASS_NAMES

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"{'=' * 60}")

            if self.config.FREEZE_BACKBONE and epoch == self.config.UNFREEZE_EPOCH:
                self._unfreeze_backbone()

            train_loss, train_labels, train_preds = self._run_epoch(self.train_loader, training=True)
            val_loss, val_labels, val_preds = self._run_epoch(self.val_loader, training=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.history['learning_rates'].append(current_lrs)

            train_metrics = print_metrics("Train", train_loss, train_labels, train_preds, class_names)
            val_metrics = print_metrics("Val", val_loss, val_labels, val_preds, class_names)

            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            if len(current_lrs) > 1:
                print(f"LR: Backbone={current_lrs[0]:.2e}, Head={current_lrs[1]:.2e}")
            else:
                print(f"LR: {current_lrs[0]:.2e}")

            is_best, current_value = self._is_new_best(val_metrics, best_value)
            if is_best:
                best_value = current_value

            self.save_checkpoint(epoch, is_best=is_best)

            early_stop_value = val_metrics['f1_macro'] if metric == 'f1_macro' else val_loss
            self.early_stopping(early_stop_value)

            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        history_path = self.config.SAVE_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"\nTraining history saved to {history_path}")


def main():
    config = X3DConfig()
    trainer = X3DTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()