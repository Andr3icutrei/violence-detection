import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class MobileNetV2_Attention(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2_Attention, self).__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        embed_dim = 1280
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        features = self.feature_extractor(x)
        features = self.pool(features)
        features = features.view(b, t, -1)
        attn_output, _ = self.attention(features, features, features)
        pooled_output = torch.mean(attn_output, dim=1)
        logits = self.classifier(pooled_output)
        return logits


class RLVS_VideoDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, num_frames=16):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                print(f"Warning: video reported 0 frames, skipping read: {video_path}")
                frame_indices = np.zeros(self.num_frames, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)

                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        finally:
            cap.release()

        video_tensor = torch.stack(frames)
        return video_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(dataset_dir):
    classes = {"Violence": 1, "NonViolence": 0}
    all_paths = []
    all_labels = []

    for cls_name, cls_label in classes.items():
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Warning: class directory not found: {cls_dir}")
            continue
        for file in os.listdir(cls_dir):
            if file.endswith(('.mp4', '.avi')):
                all_paths.append(os.path.join(cls_dir, file))
                all_labels.append(cls_label)

    video_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 / 3), stratify=y_temp, random_state=42
    )

    train_dataset = RLVS_VideoDataset(X_train, y_train, transform=video_transform, num_frames=16)
    val_dataset   = RLVS_VideoDataset(X_val,   y_val,   transform=video_transform, num_frames=16)
    test_dataset  = RLVS_VideoDataset(X_test,  y_test,  transform=video_transform, num_frames=16)

    # FIX: paper specifies batch_size=8, not 16
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(dataset_dir="../../Datasets/RLVS")

    model = MobileNetV2_Attention(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=5, min_lr=0.00001
    )

    num_epochs = 50
    early_stopping_patience = 15
    epochs_no_improve = 0
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * train_correct / train_total:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * val_correct / val_total:.2f}%")

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'saves/best_mobilenetv2_attention.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"Val loss did not improve for {early_stopping_patience} epochs."
                )
                break

    print("Training complete!")


if __name__ == '__main__':
    train_model()