import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json
from pathlib import Path

from model import X3DViolence
from dataset import X3DFlowDataset
from config import X3DConfig


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


class X3DTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = X3DViolence(
            num_classes=2,
            pretrained=config.USE_PRETRAINED,
            dropout_p=config.DROPOUT_P,
            x3d_version=config.X3D_VERSION,
            input_channels=config.INPUT_CHANNELS
        ).to(self.device)

        class_weights = torch.tensor([0.78, 1.36]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )
        self._setup_optimizer()

        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        self.train_loader = self._create_dataloader(training=True)
        self.val_loader = self._create_dataloader(training=False)

        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

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

        if self.config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ], weight_decay=self.config.WEIGHT_DECAY, betas=self.config.BETAS, eps=self.config.EPS)
        else:
            self.optimizer = optim.SGD([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ], momentum=0.9, weight_decay=self.config.WEIGHT_DECAY)

        if self.config.USE_SCHEDULER:
            if self.config.SCHEDULER_TYPE == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=self.config.T_0, T_mult=self.config.T_MULT, eta_min=self.config.ETA_MIN
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def _create_dataloader(self, training):
        dataset = X3DFlowDataset(
            dataset_info=self.config.VIOLENCE_PATH,
            num_frames=self.config.NUM_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            split_ratio=self.config.SPLIT_RATIO,
            training=training,
            augment=True,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            seed=42
        )

        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=training,
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.ACCUMULATION_STEPS

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.config.ACCUMULATION_STEPS * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                {'loss': f'{loss.item() * self.config.ACCUMULATION_STEPS:.4f}', 'acc': f'{100 * correct / total:.2f}%'}
            )

        if (batch_idx + 1) % self.config.ACCUMULATION_STEPS != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return running_loss / total, correct / total

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / total, correct / total

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filename = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)
        if is_best:
            torch.save(checkpoint, self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_best.pth")

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            if self.config.FREEZE_BACKBONE and epoch == self.config.UNFREEZE_EPOCH:
                print("Unfreezing backbone")
                for param in self.model.parameters():
                    param.requires_grad = True
                self._setup_optimizer()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            if self.scheduler:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append([g['lr'] for g in self.optimizer.param_groups])

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print("New best model saved!")

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        with open(self.config.SAVE_DIR / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=4)