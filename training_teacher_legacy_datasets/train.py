from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import R3DTransferConfig
from dataset import VideoSequenceDataset
from model import R3D18Violence


class EarlyStopping:
    """Track validation loss and signal when training should stop early."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        """Initialize early stopping state."""
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: float | None = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        """Update the early stopping state using the latest validation loss."""
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
    """Train and validate the R3D-18 violence classification model."""

    def __init__(self, config: R3DTransferConfig) -> None:
        """Create the model, optimizer, loaders, early stopping, and metric history."""
        self.config: R3DTransferConfig = config
        self.device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        dropout_p: float = getattr(config, "DROPOUT_P", 0.5)
        self.model: R3D18Violence = R3D18Violence(
            num_classes=2,
            pretrained=config.USE_PRETRAINED,
            freeze_layers=config.FREEZE_LAYERS,
            dropout_p=dropout_p,
        ).to(self.device)

        label_smoothing: float = getattr(config, "LABEL_SMOOTHING", 0.0)
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer: optim.Optimizer
        self.scheduler = None

        self._setup_optimizer()

        self.train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = self._create_dataloader(training=True)
        self.val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = self._create_dataloader(training=False)
        self.early_stopping: EarlyStopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        self.history: dict[str, list[float] | list[list[float]]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

    def _setup_optimizer(self) -> None:
        """Create the optimizer and optional learning-rate scheduler."""
        backbone_params: list[torch.nn.Parameter] = []
        head_params: list[torch.nn.Parameter] = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "fc" in name:
                head_params.append(parameter)
            else:
                backbone_params.append(parameter)

        optimizer_type: str = getattr(self.config, "OPTIMIZER", "adamw").lower()
        parameter_groups: list[dict[str, list[torch.nn.Parameter] | float]] = [
            {"params": backbone_params, "lr": self.config.BACKBONE_LR},
            {"params": head_params, "lr": self.config.HEAD_LR},
        ]

        if optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                parameter_groups,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=getattr(self.config, "BETAS", (0.9, 0.999)),
                eps=getattr(self.config, "EPS", 1e-8),
            )
        else:
            self.optimizer = optim.SGD(
                parameter_groups,
                momentum=getattr(self.config, "MOMENTUM", 0.9),
                weight_decay=self.config.WEIGHT_DECAY,
            )

        if not getattr(self.config, "USE_SCHEDULER", False):
            self.scheduler = None
            return

        scheduler_type: str = getattr(self.config, "SCHEDULER_TYPE", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=getattr(self.config, "T_0", 10),
                T_mult=getattr(self.config, "T_MULT", 2),
                eta_min=getattr(self.config, "ETA_MIN", 1e-7),
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == "reduce_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=5,
            )
        else:
            self.scheduler = None

    def _create_dataloader(self, training: bool) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create a PyTorch DataLoader for the requested split."""
        if self.config.DATASET_NAME == "Mix":
            violence_paths, non_violence_paths = self.config.get_mix_paths()
        else:
            violence_paths = self.config.VIOLENCE_PATH
            non_violence_paths = self.config.NON_VIOLENCE_PATH

        dataset: VideoSequenceDataset = VideoSequenceDataset(
            violence_path=violence_paths,
            non_violence_path=non_violence_paths,
            n_frames=self.config.N_FRAMES,
            split_ratio=self.config.SPLIT_RATIO,
            training=training,
            augment=True,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=training,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
        )

    def train_epoch(self) -> tuple[float, float]:
        """Run one training epoch and return average loss and accuracy."""
        self.model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(outputs, labels)
            loss.backward()

            if hasattr(self.config, "GRAD_CLIP"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            self.optimizer.step()

            batch_size: int = inputs.size(0)
            predicted: torch.Tensor = torch.max(outputs.data, 1)[1]
            running_loss += float(loss.item()) * batch_size
            total += int(labels.size(0))
            correct += int((predicted == labels).sum().item())

        epoch_loss: float = running_loss / total if total > 0 else 0.0
        epoch_accuracy: float = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_accuracy

    def validate_epoch(self) -> tuple[float, float]:
        """Run one validation epoch and return average loss and accuracy."""
        self.model.eval()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = self.criterion(outputs, labels)
                batch_size: int = inputs.size(0)
                predicted: torch.Tensor = torch.max(outputs.data, 1)[1]

                running_loss += float(loss.item()) * batch_size
                total += int(labels.size(0))
                correct += int((predicted == labels).sum().item())

        epoch_loss: float = running_loss / total if total > 0 else 0.0
        epoch_accuracy: float = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Persist a checkpoint for the current model, optimizer, scheduler, and history state."""
        checkpoint: dict[str, int | dict | list] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        filename: Path = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)

        if is_best:
            best_filename: Path = self.config.SAVE_DIR / f"{self.config.MODEL_NAME}_best.pth"
            torch.save(checkpoint, best_filename)

    def train(self) -> dict[str, list[float] | list[list[float]]]:
        """Run the full training loop and return the metric history."""
        best_val_loss: float = float("inf")

        for epoch in range(self.config.NUM_EPOCHS):
            if epoch == self.config.UNFREEZE_EPOCH:
                self.model.unfreeze_all()
                self._setup_optimizer()

            train_loss: float
            train_accuracy: float
            val_loss: float
            val_accuracy: float
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lrs: list[float] = [float(group["lr"]) for group in self.optimizer.param_groups]
            self.history["learning_rates"].append(current_lrs)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_accuracy)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_accuracy)

            is_best: bool = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                break

        history_path: Path = self.config.SAVE_DIR / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)

        return self.history


def main() -> None:
    """Train the model with the default configuration."""
    config: R3DTransferConfig = R3DTransferConfig()
    trainer: R3D18Trainer = R3D18Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()