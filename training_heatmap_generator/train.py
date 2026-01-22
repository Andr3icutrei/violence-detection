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

        self.model = R3D18Violence(
            num_classes=2,
            pretrained=config.USE_PRETRAINED,
            freeze_layers=config.FREEZE_LAYERS
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self._setup_optimizer()

        self.train_loader = self._create_dataloader(training=True)
        self.val_loader = self._create_dataloader(training=False)

        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
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

        self.optimizer = optim.SGD([
            {'params': backbone_params, 'lr': self.config.BACKBONE_LR},
            {'params': head_params, 'lr': self.config.HEAD_LR}
        ], momentum=self.config.MOMENTUM, weight_decay=self.config.WEIGHT_DECAY)

    def _create_dataloader(self, training):
        dataset = VideoSequenceDataset(
            violence_path=self.config.VIOLENCE_PATH,
            non_violence_path=self.config.NON_VIOLENCE_PATH,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=training,
            augment=True,
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
                print("Unfreezing all layers")
                self.model.unfreeze_all()
                self._setup_optimizer()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        history_path = self.config.SAVE_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


def main():
    config = R3DTransferConfig()

    print(f"Device: {config.DEVICE}")
    print(f"Training R3D-18 with transfer learning on {config.DATASET_NAME} dataset")
    print(f"Pretrained: {config.USE_PRETRAINED}")
    print(f"Frozen layers: {config.FREEZE_LAYERS}")

    trainer = R3D18Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()