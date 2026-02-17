import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json
from pathlib import Path
from collections import Counter

from model import SlowFastViolence
from dataset import SlowFastVideoDataset
from config import SlowFastConfig
from focal_loss import FocalLoss
from balanced_sampler import create_balanced_sampler


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class SlowFastTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = SlowFastViolence(
            num_classes=config.NUM_CLASSES,
            pretrained=config.USE_PRETRAINED,
            dropout_p=config.DROPOUT_P,
            slowfast_alpha=config.SLOWFAST_ALPHA,
            slowfast_beta=config.SLOWFAST_BETA
        ).to(self.device)

        self.train_loader = self._create_dataloader(training=True)
        self.val_loader = self._create_dataloader(training=False)

        class_weights = None
        if config.USE_CLASS_WEIGHTS:
            class_weights = self._calculate_class_weights()
            print(f"Class weights: {class_weights}")

        if config.USE_FOCAL_LOSS:
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=config.FOCAL_GAMMA,
                label_smoothing=config.LABEL_SMOOTHING
            )
            print(f"Using Focal Loss with gamma={config.FOCAL_GAMMA}")
        else:
            if class_weights is not None:
                class_weights = torch.tensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.LABEL_SMOOTHING
            )
            print("Using Cross Entropy Loss")

        self._setup_optimizer()

        self.scaler = GradScaler() if config.USE_AMP else None
        if config.USE_AMP:
            print("Using Automatic Mixed Precision (AMP)")

        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

    def _calculate_class_weights(self):
        all_labels = self.train_loader.dataset.labels

        label_counts = Counter(all_labels)

        total_samples = len(all_labels)
        num_classes = self.config.NUM_CLASSES

        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            weights.append(weight)

        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        return weights

    def _setup_optimizer(self):
        if self.config.FREEZE_BACKBONE:
            for name, param in self.model.named_parameters():
                if 'blocks.5' not in name:
                    param.requires_grad = False

        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'blocks.5' in name or 'proj' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer_type = self.config.OPTIMIZER.lower()

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ],
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS,
                eps=self.config.EPS)
        else:
            self.optimizer = optim.SGD([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ],
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY)

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
                    patience=5,
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def _create_dataloader(self, training):
        violence_path = self.config.VIOLENCE_PATH
        non_violence_path = self.config.NON_VIOLENCE_PATH

        dataset = SlowFastVideoDataset(
            violence_path=violence_path,
            non_violence_path=non_violence_path,
            slow_frames=self.config.SLOW_FRAMES,
            fast_frames=self.config.FAST_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            slowfast_alpha=self.config.SLOWFAST_ALPHA,
            slowfast_beta=self.config.SLOWFAST_BETA,
            split_ratio=self.config.SPLIT_RATIO,
            training=training,
            augment=True,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP
        )

        sampler = None
        shuffle = training

        if training and self.config.USE_BALANCED_SAMPLING:
            sampler = create_balanced_sampler(dataset)
            shuffle = False
            print("Using balanced sampling for training")

        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        self.optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(pbar):
            slow_inputs = inputs[0].to(self.device)
            fast_inputs = inputs[1].to(self.device)
            labels = labels.to(self.device)

            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model([slow_inputs, fast_inputs])
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config.ACCUMULATION_STEPS

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model([slow_inputs, fast_inputs])
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.ACCUMULATION_STEPS
                loss.backward()

            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                if self.config.USE_AMP:
                    self.scaler.unscale_(self.optimizer)
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

            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.ACCUMULATION_STEPS:.4f}',
                'acc': f'{100 * correct / total:.2f}%',
                'eff_bs': self.config.EFFECTIVE_BATCH_SIZE
            })

        if (batch_idx + 1) % self.config.ACCUMULATION_STEPS != 0:
            if self.config.USE_AMP:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.optimizer.step()

            self.optimizer.zero_grad()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in pbar:
                slow_inputs = inputs[0].to(self.device)
                fast_inputs = inputs[1].to(self.device)
                labels = labels.to(self.device)

                if self.config.USE_AMP:
                    with autocast():
                        outputs = self.model([slow_inputs, fast_inputs])
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model([slow_inputs, fast_inputs])
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        filename = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)

        if is_best:
            best_filename = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_best.pth"
            torch.save(checkpoint, best_filename)

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 50)

            if self.config.FREEZE_BACKBONE and epoch == self.config.UNFREEZE_EPOCH:
                print("Unfreezing backbone")
                for param in self.model.parameters():
                    param.requires_grad = True
                self._setup_optimizer()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.history['learning_rates'].append(current_lrs)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")
            if len(current_lrs) > 1:
                print(f"Learning Rates: Backbone={current_lrs[0]:.2e}, Head={current_lrs[1]:.2e}")
            else:
                print(f"Learning Rate: {current_lrs[0]:.2e}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model saved!")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        history_path = self.config.SAVE_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"\nTraining history saved to {history_path}")


def main():
    config = SlowFastConfig()

    print(f"Device: {config.DEVICE}")
    print(f"Training SlowFast on {config.DATASET_NAME} dataset")
    print(f"Pretrained: {config.USE_PRETRAINED}")
    print(f"Slow Frames: {config.SLOW_FRAMES}, Fast Frames: {config.FAST_FRAMES}")
    print(f"Alpha: {config.SLOWFAST_ALPHA}, Beta: {config.SLOWFAST_BETA}")
    print(f"Dropout: {config.DROPOUT_P}")
    print(f"Label Smoothing: {config.LABEL_SMOOTHING}")
    print(f"Backbone LR: {config.BACKBONE_LR}, Head LR: {config.HEAD_LR}")
    if config.USE_SCHEDULER:
        print(f"Scheduler: {config.SCHEDULER_TYPE}")
    if config.USE_AMP:
        print("AMP: Enabled")

    trainer = SlowFastTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()