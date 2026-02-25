import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import json
from collections import Counter
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score

from model import SlowFastViolence
from dataset import SlowFastVideoDataset
from config import SlowFastConfig
from focal_loss import FocalLoss
from balanced_sampler import create_balanced_sampler

FROZEN_BLOCKS = [0, 1, 2]
UNFREEZE_EPOCH = 20


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


def _compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def _print_epoch_metrics(split_name, loss, all_labels, all_preds, class_names):
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    print(f"\n{split_name} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | F1 macro: {f1_macro:.4f} | Recall macro: {recall_macro:.4f}")

    header = f"{'Class':<22} {'F1':>8} {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        f1_val = f1_per_class[i] if i < len(f1_per_class) else 0.0
        rec_val = recall_per_class[i] if i < len(recall_per_class) else 0.0
        print(f"{name:<22} {f1_val:>8.4f} {rec_val:>8.4f}")

    print("\nConfusion Matrix:")
    col_width = 6
    header_row = " " * 22 + "".join(f"{n[:col_width]:>{col_width}}" for n in class_names)
    print(header_row)
    for i, name in enumerate(class_names):
        row = f"{name:<22}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(class_names)))
        print(row)
    print()


def _freeze_early_blocks(model):
    for name, param in model.named_parameters():
        if any(f'backbone.blocks.{i}.' in name for i in FROZEN_BLOCKS):
            param.requires_grad = False


def _unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


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

        if config.FREEZE_BACKBONE:
            _freeze_early_blocks(self.model)
            frozen_names = [
                n for n, p in self.model.named_parameters() if not p.requires_grad
            ]
            print(f"Frozen {len(frozen_names)} parameter tensors (backbone blocks 0, 1, 2)")
            print(f"Training blocks 3, 4, 5 until epoch {UNFREEZE_EPOCH}")

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
            'train_f1': [], 'val_f1': [],
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
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
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
            seed=self.config.SEED,
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

    def train_epoch(self, pbar):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        last_grad_norm = 0.0

        all_preds = []
        all_labels = []

        self.optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            slow_inputs = inputs[0].to(self.device)
            fast_inputs = inputs[1].to(self.device)
            labels = labels.to(self.device)

            if self.config.USE_AMP:
                with autocast(device_type='cuda'):
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

            running_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            running_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            pbar.set_postfix({
                'phase': 'train',
                'loss': f'{loss.item() * self.config.ACCUMULATION_STEPS:.4f}',
                'acc': f'{100 * correct / total:.2f}%',
                'f1': f'{running_f1:.4f}',
                'recall': f'{running_recall:.4f}',
                'grad_norm': f'{last_grad_norm:.2f}'
            })
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

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        _print_epoch_metrics("TRAIN", epoch_loss, all_labels, all_preds, self.config.CLASS_NAMES)

        return epoch_loss, epoch_acc, all_labels, all_preds

    def validate_epoch(self, pbar):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                slow_inputs = inputs[0].to(self.device)
                fast_inputs = inputs[1].to(self.device)
                labels = labels.to(self.device)

                if self.config.USE_AMP:
                    with autocast(device_type='cuda'):
                        outputs = self.model([slow_inputs, fast_inputs])
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model([slow_inputs, fast_inputs])
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                running_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                running_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

                pbar.set_postfix({
                    'phase': 'val',
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%',
                    'f1': f'{running_f1:.4f}',
                    'recall': f'{running_recall:.4f}'
                })
                pbar.update(1)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        _print_epoch_metrics("VAL", epoch_loss, all_labels, all_preds, self.config.CLASS_NAMES)

        return epoch_loss, epoch_acc, all_labels, all_preds

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

            if self.config.FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
                print(f"Epoch {epoch + 1}: unfreezing stem, layer1, layer2 (blocks 0, 1, 2)")
                self.config.FREEZE_BACKBONE = False
                _unfreeze_all(self.model)
                self._setup_optimizer()
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"Trainable parameters after unfreeze: {trainable:,}")

            total_batches = len(self.train_loader) + len(self.val_loader)
            pbar = tqdm(total=total_batches, desc=f"Epoch {epoch + 1}")

            train_loss, train_acc, train_labels, train_preds = self.train_epoch(pbar)
            val_loss, val_acc, val_labels, val_preds = self.validate_epoch(pbar)

            pbar.close()

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.history['learning_rates'].append(current_lrs)

            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

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