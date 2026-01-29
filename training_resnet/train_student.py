import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import sys

sys.path.append('/mnt/project')

from student_model import R3D18Student
from smart_crop_dataset import SmartCropDataset
from dataset import VideoSequenceDataset
from config import R3DTransferConfig


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
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


class StudentTrainer:
    def __init__(self, config, teacher_model_path):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = R3D18Student(
            num_classes=2,
            pretrained=True,
            dropout_p=config.DROPOUT_P
        ).to(self.device)

        self.teacher_model_path = teacher_model_path

        self._setup_data()
        self._setup_training()

    def _setup_data(self):
        if self.config.DATASET_NAME == 'Mix':
            violence_paths, non_violence_paths = self.config.get_mix_paths()
        else:
            violence_paths = self.config.VIOLENCE_PATH
            non_violence_paths = self.config.NON_VIOLENCE_PATH

        self.train_dataset = SmartCropDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            teacher_model_path=self.teacher_model_path,
            config=self.config,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=True,
            smart_crop_prob=self.config.SMART_CROP_PROB,
            threshold=self.config.SMART_CROP_THRESHOLD
        )

        self.val_dataset = VideoSequenceDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=False,
            augment=False,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )

    def _setup_training(self):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.LABEL_SMOOTHING)

        if self.config.OPTIMIZER.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.BACKBONE_LR,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS,
                eps=self.config.EPS
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.BACKBONE_LR,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )

        if self.config.USE_SCHEDULER:
            if self.config.SCHEDULER_TYPE == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.config.T_0,
                    T_mult=self.config.T_MULT,
                    eta_min=self.config.ETA_MIN
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

        self.early_stopping = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Training")
        for batch_idx, (inputs, labels) in enumerate(pbar):
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

        train_loss = running_loss / total
        train_acc = correct / total


        return train_loss, train_acc

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation")
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        return val_loss, val_acc

    def train(self):
        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = self.config.SAVE_DIR / f"student_r3d18_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'history': history
                }, save_path)
                print(f"Model saved to {save_path}")

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        history_path = self.config.SAVE_DIR / "student_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    config = R3DTransferConfig(dataset_name='Hockey', use_smart_crop=True)

    teacher_model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not teacher_model_path.exists():
        raise FileNotFoundError(f"Teacher model not found at {teacher_model_path}")

    trainer = StudentTrainer(config, teacher_model_path)
    trainer.train()