import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path

from model import R3D18Violence
from dataset import VideoSequenceDataset
from config import R3DTransferConfig


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


class R3D18Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        dropout_p = getattr(config, 'DROPOUT_P', 0.5)
        self.model = R3D18Violence(
            num_classes=2,
            pretrained=config.USE_PRETRAINED,
            freeze_layers=config.FREEZE_LAYERS,
            dropout_p=dropout_p
        ).to(self.device)

        label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self._setup_optimizer()

        self.train_loader = self._create_dataloader(training=True)
        self.val_loader = self._create_dataloader(training=False)

        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

    def _setup_optimizer(self):
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer_type = getattr(self.config, 'OPTIMIZER', 'adamw').lower()

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ],
                weight_decay=self.config.WEIGHT_DECAY,
                betas=getattr(self.config, 'BETAS', (0.9, 0.999)),
                eps=getattr(self.config, 'EPS', 1e-8))
        else:
            self.optimizer = optim.SGD([
                {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
                {'params': head_params, 'lr': self.config.HEAD_LR}
            ],
                momentum=getattr(self.config, 'MOMENTUM', 0.9),
                weight_decay=self.config.WEIGHT_DECAY)

        if getattr(self.config, 'USE_SCHEDULER', False):
            scheduler_type = getattr(self.config, 'SCHEDULER_TYPE', 'cosine')

            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=getattr(self.config, 'T_0', 10),
                    T_mult=getattr(self.config, 'T_MULT', 2),
                    eta_min=getattr(self.config, 'ETA_MIN', 1e-7)
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
                    verbose=True
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def _create_dataloader(self, training):
        violence_path = self.config.VIOLENCE_PATH
        non_violence_path = self.config.NON_VIOLENCE_PATH

        dataset = VideoSequenceDataset(
            violence_path=violence_path,
            non_violence_path=non_violence_path,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=training,
            augment=training,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD
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
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()

            if hasattr(self.config, 'GRAD_CLIP'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
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

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
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

            if epoch == self.config.UNFREEZE_EPOCH:
                self.model.unfreeze_all()
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
            print(f"Learning Rates: Backbone={current_lrs[0]:.2e}, Head={current_lrs[1]:.2e}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                break

        history_path = self.config.SAVE_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


def main():
    config = R3DTransferConfig(dataset_name='Mix')

    config.VIOLENCE_PATH = Path("../../Datasets/Mix_SmartCropped/Violence")
    config.NON_VIOLENCE_PATH = Path("../../Datasets/Mix_SmartCropped/NonViolence")

    config.SAVE_DIR = Path("checkpoints_r3d18_mix_smart_crop")
    config.MODEL_NAME = "r3d18_violence_mix_smart_crop"
    config.SAVE_DIR.mkdir(exist_ok=True, parents=True)

    config.NUM_EPOCHS = 50
    config.BATCH_SIZE = 32

    trainer = R3D18Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()