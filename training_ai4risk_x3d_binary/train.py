from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import X3DConfig
from dataset import DatasetItem, X3DVideoDataset
from model import X3DViolence


logger: logging.Logger = logging.getLogger(__name__)


class EarlyStopping:
    """Tracks validation loss and signals when training should stop."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        """Initializes early stopping state."""

        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        """Updates the early stopping state from the latest validation loss."""

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
    """Builds the X3D model, dataloaders, optimizer, scheduler, and training loop."""

    def __init__(self, config: X3DConfig, verbose: bool = True) -> None:
        """Initializes model training components from the provided configuration."""

        self.config: X3DConfig = config
        self.verbose: bool = verbose
        self.device: torch.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model: X3DViolence = X3DViolence(
            num_classes=2,
            pretrained=config.USE_PRETRAINED,
            dropout_p=config.DROPOUT_P,
            x3d_version=config.X3D_VERSION,
        ).to(self.device)

        class_weights: torch.Tensor = torch.tensor([0.78, 1.36], dtype=torch.float32, device=self.device)
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING,
        )
        self.optimizer: optim.Optimizer
        self.scheduler: Optional[LRScheduler | ReduceLROnPlateau]
        self._setup_optimizer()

        self.train_loader: DataLoader[DatasetItem] = self._create_dataloader(training=True)
        self.val_loader: DataLoader[DatasetItem] = self._create_dataloader(training=False)
        self.early_stopping: EarlyStopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        self.history: dict[str, list[float] | list[list[float]]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

    def _setup_optimizer(self) -> None:
        """Configures trainable parameters, optimizer, and optional scheduler."""

        if self.config.FREEZE_BACKBONE:
            for name, param in self.model.named_parameters():
                if "blocks.5" not in name:
                    param.requires_grad = False

        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "blocks.5" in name or "proj" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer_type: str = self.config.OPTIMIZER.lower()
        parameter_groups: list[dict[str, list[nn.Parameter] | float]] = [
            {"params": backbone_params, "lr": self.config.BACKBONE_LR},
            {"params": head_params, "lr": self.config.HEAD_LR},
        ]

        if optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                parameter_groups,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS,
                eps=self.config.EPS,
            )
        else:
            self.optimizer = optim.SGD(
                parameter_groups,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY,
            )

        if not self.config.USE_SCHEDULER:
            self.scheduler = None
            return

        scheduler_type: str = self.config.SCHEDULER_TYPE

        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_MULT,
                eta_min=self.config.ETA_MIN,
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        elif scheduler_type == "reduce_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=5,
            )
        else:
            self.scheduler = None

    def _create_dataloader(self, training: bool) -> DataLoader[DatasetItem]:
        """Creates the dataloader for either the training or validation split."""

        if self.config.VIOLENCE_PATH is None or self.config.NON_VIOLENCE_PATH is None:
            raise ValueError("Dataset paths must be configured before creating dataloaders.")

        dataset: X3DVideoDataset = X3DVideoDataset(
            violence_path=self.config.VIOLENCE_PATH,
            non_violence_path=self.config.NON_VIOLENCE_PATH,
            num_frames=self.config.NUM_FRAMES,
            temporal_stride=self.config.TEMPORAL_STRIDE,
            split_ratio=self.config.SPLIT_RATIO,
            training=training,
            augment=True,
            mean=self.config.KINETICS_MEAN,
            std=self.config.KINETICS_STD,
            crop_size=self.config.CROP_SIZE,
            use_crop=self.config.USE_CROP,
        )

        loader: DataLoader[DatasetItem] = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=training,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
        )
        return loader

    def train_epoch(self) -> tuple[float, float]:
        """Runs one training epoch and returns average loss and accuracy."""

        self.model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        last_batch_idx: int = -1
        progress_bar: tqdm = tqdm(self.train_loader, desc="Training", disable=not self.verbose, leave=False)

        self.optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            last_batch_idx = batch_idx
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(outputs, labels)
            scaled_loss: torch.Tensor = loss / self.config.ACCUMULATION_STEPS
            scaled_loss.backward()

            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                if hasattr(self.config, "GRAD_CLIP"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_size: int = labels.size(0)
            predictions: torch.Tensor = torch.argmax(outputs.detach(), dim=1)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += (predictions == labels).sum().item()

        if last_batch_idx >= 0 and (last_batch_idx + 1) % self.config.ACCUMULATION_STEPS != 0:
            if hasattr(self.config, "GRAD_CLIP"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if total == 0:
            return 0.0, 0.0

        epoch_loss: float = running_loss / total
        epoch_acc: float = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> tuple[float, float]:
        """Runs one validation epoch and returns average loss and accuracy."""

        self.model.eval()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            progress_bar: tqdm = tqdm(self.val_loader, desc="Validation", disable=not self.verbose, leave=False)

            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = self.criterion(outputs, labels)
                batch_size: int = labels.size(0)
                predictions: torch.Tensor = torch.argmax(outputs, dim=1)

                running_loss += loss.item() * batch_size
                total += batch_size
                correct += (predictions == labels).sum().item()

        if total == 0:
            return 0.0, 0.0

        epoch_loss: float = running_loss / total
        epoch_acc: float = correct / total
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Saves the model, optimizer, scheduler, and training history."""

        checkpoint: dict = {
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

    def train(self) -> None:
        """Runs the full training process, saves checkpoints, and writes training history."""

        best_val_loss: float = float("inf")

        for epoch in range(self.config.NUM_EPOCHS):
            if self.config.FREEZE_BACKBONE and epoch == self.config.UNFREEZE_EPOCH:
                for param in self.model.parameters():
                    param.requires_grad = True
                self._setup_optimizer()

            train_loss: float
            train_acc: float
            val_loss: float
            val_acc: float

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lrs: list[float] = [float(group["lr"]) for group in self.optimizer.param_groups]
            self.history["learning_rates"].append(current_lrs)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            is_best: bool = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            if self.verbose:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | train_acc=%.2f%% | val_loss=%.4f | val_acc=%.2f%% | best=%s",
                    epoch + 1,
                    self.config.NUM_EPOCHS,
                    train_loss,
                    train_acc * 100,
                    val_loss,
                    val_acc * 100,
                    is_best,
                )

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                if self.verbose:
                    logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        history_path: Path = self.config.SAVE_DIR / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)

        if self.verbose:
            logger.info("Training history saved to %s.", history_path)


def main() -> None:
    """Runs model training with the default configuration."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config: X3DConfig = X3DConfig()
    trainer: X3DTrainer = X3DTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()